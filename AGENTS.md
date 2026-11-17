# AGENTS.md — emotion2vec+ 日本語感情音声FT

このリポジトリでの作業内容・構成・実行手順。emotion2vec公式コード(`upstream/`, `iemocap_downstream/`, `scripts/`)は変更せず、新規実装はすべて `finetune/` と `container/` に集約している。

## 実装内容 (2026-11-16)

### 0. クリップ実体化 — `finetune/build_clips.py`

SERのクリップを流用せず、元コーパスから**このリポジトリ内** (`data/clips/<dataset>/`) に
16kHz/16bit/mono wavをffmpegで実体化する。SER-model-comparisonのmanifestは
コーパスパス・ラベル情報の参照元としてのみ使用する。

- JTES (`/disk/fs1/Corpus/JTES`)、OGVC (`/disk/fs1/Corpus/OGVC`): manifestのsource_pathをffmpegで直接変換。
- JVNV: manifestのsource_pathの展開済みツリーは消滅しているため、
  `jvnv_ver1.zip` アーカイブから該当wavを抽出して変換(`--jvnv-zip` で指定)。
- 出力: `data/clips/<dataset>/<utt_id>.wav` + `finetune/manifests_src/<dataset>.jsonl`
  (clip_pathを自前クリップのパスに書き換えたmanifest)。

実行:
```bash
python finetune/build_clips.py  # 全4データセット、28,085クリップ
```

### 1. データ分割 — `finetune/splits.py`

`finetune/manifests_src/*.jsonl` (build_clips.pyが生成、クリップは`data/clips/`) の
JVNV / JTES / OGVC-Acted / OGVC-Natural マニフェストから、学習・評価・テスト分割を生成する。

出力: `finetune/manifests_ft/<dataset>/{train,eval,test}.jsonl` + `exclusions.{md,json}`

| データセット | 分割方法 | train / eval / test | 除外 |
| --- | --- | --- | --- |
| JVNV (1,615発話/4話者/6感情) | 発話ベース、(話者×感情)層化、seed=42、70/15/15 | 1,127 / 244 / 244 | なし |
| JTES (20,000発話/100話者/4感情 ang,joy,neu,sad) | **話者ベース(ユーザー固定)** | 18,400 / 800 / 800 | eval=f10,f29,m10,m20、test=f16,f35,m14,m22 を学習から除外(残り92話者がtrain) |
| OGVC-Acted (2,656発話/4話者/8感情) | 発話ベース、(話者×感情)層化、seed=42 | 1,416 / 300 / 300 | 640行: ACC/ANT(感情ラベルとして9クラスに非該当)。intensity=0はNEUとして使用 |
| OGVC-Natural (3,814発話/13話者/3評定者) | 話者ベース(greedy比率fit、seed=42) | 1,789 / 493 / 617 | 915行: 多数決ラベルがACC/ANT/OTHの行。eval話者=03_FTY,06_FWA、test話者=04_MNN,04_MSY |

仕様:
- **ラベル空間はemotion2vec+の9クラス**(ANG, DIS, FEA, HAP, NEU, SAD, SUR, OTHER, UNKNOWN)。
  学習データに出現するのは7クラス(OTHER/UNKNOWNは学習データに存在しない。OGVCのOTHは非感情として除外)。
- **OGVC-Naturalの多数決**: 3評定者の多数決(>=2/3)でラベルが決定しない発話は省く。
  上流manifestではgold(多数決>=2/3)とallagreementに既に分離済み。本スクリプトはratersから再検証し、
  多数決非決定かつgoldに入っている行が0件であることを確認済み(検証スクリプム参照)。
- **どの話者・発話を除外したか**は `exclusions.md`(人間可読)と `exclusions.json`(機械可読)に全件記録。
  JTESは話者割当、OGVC-Naturalは話者割当+除外行(utt_id, speaker, native_emotion, reason, raters)を記載。

実行:
```bash
python finetune/splits.py  # manifests_src(自前クリップ) -> manifests_ft
```

### 2. FTモデル — `finetune/model.py`

- funasr `AutoModel` で emotion2vec+ をロード(`init_param=model.pt`, `scope_map: d2v_model.->none` は
  モデル同梱のconfig.yamlから適用)。FT元: `emotion2vec_plus_base` (depth 8, embed_dim 768, proj 768->9)。
- **学習層: 上位層のみ**(ユーザー決定 2026-11-16)。
  - base: depth 8 → **blocks.5,6,7(第6〜8層)+ proj** を学習(`--train-blocks 3`)
  - large: depth 24 → **最後の4層(blocks.20-23)+ proj**(`--train-blocks 4`)
  - この設定の根拠: emotion2vec+公式FTはconvフロントエンドに勾配を流さない(`feature_grad_mult=0.0`)、
    wav2vec2/HuBERT系FTの標準手法も下層凍結+上位数層のみ学習。
  - 当初「layer8~11」としていたが、emotion2vec+ baseの実層数は12でなく**8**であることをチェックポイント
    実検証で確認し、ユーザーの指示「8層しかないなら6-8層を学習」に従い blocks.5-7 に確定した。
- `freeze_top_layers()` は `modality_encoders.AUDIO`(conv frontend)と下位blockを全freezeし、
  パラメータ監査(dict)を返す。検証済み: trainable=21,270,537 (blocks.5-7+projのみ)、
  逆伝播で勾配が流れるのも同パラメータのみ(フロントエンド勾配ゼロ)。
- `forward_logits()`: extract_features → 非パディングフレームの平均プーリング → proj → (B,9) logits。

### 3. データセット/学習ループ — `finetune/dataset.py`, `finetune/train.py`

- 16kHz wavを読み、`F.layer_norm`で正規化(emotion2vec+)の前処理と同一)、可変長パディング+padding_mask。
- CrossEntropyLoss + AdamW(trainableのみ) + warmup付き線形decay + grad clip 5.0。
- 10ステップごとにeval_loss/WAを計測、bestを `best_model.pt` (modelのstate_dict)に保存。
  funasr AutoModelと同じ形式で再ロード可能。
- スモークテストモード: `--max-steps N --limit M`(データセット先頭M行でNステップだけ学習)。

### 4. 設定 — `finetune/config/base.yaml`, `finetune/config/large.yaml`

base: train_blocks=3, batch_size=16, lr=1e-5, epochs=10。
large: train_blocks=4, batch_size=8, lr=5e-6(初回ロード時にparam_audit.jsonでdepthを確認すること)。

### 5. コンテナ — `container/apptainer.def`

- ベース: `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`、python3.12 venv、torch 2.15.0+cu132、
  funasr 1.4.24、soundfile、omegaconf、hydra-core、pyyaml、tensorboard。
  (ubuntu24.04(noble)にはpython3.11が無いため3.12を使用。torch 2.15.0+cu132はcp312 wheelあり。
  ffmpeg/gitは不使用: クリップは事前生成済み。root-mapped namespaceでdpkg postinstの
  chownが失敗するため、fontconfigを引き込む重いパッケージを避けている)
- `%runscript`: `apptainer run --nv e2v-ft.sif finetune/config/base.yaml` で学習開始。
  (`%test`はGPU前提でビルドマシンを選ぶため置いていない。検証は実行後に手動で行う)

ビルドと実行(学習マシン):
```bash
cd container
# ビルド(既存apptainer.sifがある場合・defを編集した場合):
apptainer build --fakeroot apptainer.sif apptainer.def

# 実行: モデル・クリップ・リポジトリをbindする。クリップはmanifestのclip_pathと同じ絶対パスで見せること。
apptainer exec --nv \
  --bind /path/to/emotion2vec:/workspace/emotion2vec \
  --bind /path/to/SER-model-comparison/models:/models \
  --bind /path/to/SER-model-comparison/data/clips:/path/to/SER-model-comparison/data/clips \
  apptainer.sif /opt/venv/bin/python /workspace/emotion2vec/finetune/train.py \
    --model-dir /models/emotion2vec_plus_base \
    --manifest-root /workspace/emotion2vec/finetune/manifests_ft \
    --config /workspace/emotion2vec/finetune/config/base.yaml
```
注意: manifestのclip_pathは絶対パス(このマシンでは SER-model-comparison/data/clips 配下)なので、
学習マシンでは `--bind <クリップ置き場>:<manifestと同じ絶対パス>` で一致させること
(または分割manifestを学習マシンのパスに合わせて再生成する)。

検証済み(2026-11-17, このマシン): apptainer.sif + --nv で python 3.12.3 / torch 2.15.0+cu132 /
funasr 1.4.24 / CUDA RTX 3090 を認識し、コンテナ内で学習6ステップが完走(loss有限・変化、
best_model.pt保存)。ログ: finetune/runs/smoke_sif/train.log。

### 6. 学習マシンでの実行手順(本番)

```bash
# 0) クリップ実体化(元コーパスとSER manifestが見えるマシンで)
python finetune/build_clips.py
# 1) 分割manifest生成
python finetune/splits.py --manifest-root <SER-model-comparison>/manifests
# 2) FT実行(このマシンでスモークテスト済み: 12ステップ, loss有限・変化, ckpt保存)
python finetune/train.py --config finetune/config/base.yaml
# またはコンテナ経由(上記 apptainer run)
```

## 検証記録(このマシン)

- C1 分割: 発話重複なし・話者重複なし(話者ベース分割)・ラベル空間7クラス・JTES話者割当一致
  (eval={f10,f29,m10,m20}, test={f16,f35,m14,m22}, train=92話者)・OGVC-Natural多数決goldのcross-check。
- C2 パラメータ監査: trainable = blocks.5,6,7 + proj のみ(21,270,537)。逆伝播の非ゼロ勾配も同層のみ。
- C3 スモークテスト(RTX 3090, CUDA): 12ステップ学習、loss有限・変化(0.0003→8.19)、
  eval_loss 8.06、`finetune/runs/smoke2/best_model.pt` 保存。ログ: `finetune/runs/smoke2/train.log`
  (全4データセット、JTES含む)。
- C0 クリップ自前実体化: 28,085クリップ(16kHz/mono/PCM_16検証済み、JVNVはzip抽出)を
  data/clips/に生成、missing=0。分割再生成後、venv直・sif内の両面で学習スモーク完走
  (runs/smoke_local, runs/smoke_local_sif)。
- C4 コンテナ: apptainer.sif(cuda:12.9.1/ubuntu24.04/python3.12)をビルドし、
  `apptainer exec --nv` でCUDA(RTX 3090)認識・コンテナ内学習6ステップ完走
  (loss有限・変化、best_model.pt保存)。ログ: `finetune/runs/smoke_sif/train.log`。
