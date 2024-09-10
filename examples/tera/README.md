# TERA Speech Foundation Model - Fairseq Implementation

This repository contains code for pre-training and evaluating the [TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](https://arxiv.org/abs/2007.06028), a self-supervised speech foundation model for speech processing tasks.

## Installing Fairseq

This project relies on [Fairseq](https://github.com/facebookresearch/fairseq). To install it, run the following commands:
```bash
git clone https://github.com/andi611/fairseq.git
cd fairseq
pip install --editable ./
```
Read [here](https://github.com/andi611/fairseq/tree/master) for more details.

## Pre-training

This version of TERA is modified from the HuBERT example ([HuBERT in fairseq](https://github.com/andi611/fairseq/tree/master/examples/hubert)) with minimal modification of code. As a result, the `task.data` and `task.label_dir` parameters in the following commands are not actually used in the TERA implementation but are required due to the HuBERT-based code structure. To obtain those, please follow the instructions in the [HuBERT example directory](https://github.com/andi611/fairseq/tree/master/examples/hubert).

### _Slim_ Model

To pre-train the TERA _Slim_ model on LibriSpeech:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /work/a129195789/fairseq/examples/tera/config/pretrain/ \
  --config-name tera_slim_librispeech.yaml \
  task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

### _Slim_ Model with Different % Sizes

For the _Slim_ % model:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /work/a129195789/fairseq/examples/tera/config/pretrain/exp \
  --config-name tera_slim_200per_50flops_librispeech \
  task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

## Downstream Evaluation

### Important: S3PRL Toolkit Installation

Before proceeding with downstream evaluation, you must install the S3PRL toolkit from a specific branch. Use the following command:

```bash
git clone -b tera2 https://github.com/s3prl/s3prl.git
cd s3prl
pip install -e .
```

This will install the S3PRL toolkit from the `tera2` branch, which is required for the downstream evaluation tasks in this project.

### Converting Checkpoint Format

After pretraining, convert the checkpoint format using s3prl:

```bash
python3 upstream/tera2/convert.py \
  result/tera2_slim_960_200per/checkpoints/checkpoint_best.pt \
  --output_dir=result/tera2_slim_960_200per/converted_ckpts/
```

### Running Downstream Tasks

#### ASR (Automatic Speech Recognition)

```bash
# Training
python3 run_downstream.py -m train -d asr -u tera2_local \
  -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt \
  -n asr_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4

# Evaluation
python3 run_downstream.py -m evaluate -t "test-clean" \
  -e result/downstream/asr_tera2_slim_960_200per_1e4/dev-clean-best.ckpt
```

#### PR (Phoneme Recognition)

```bash
# Training
python3 run_downstream.py -m train -d ctc -c downstream/ctc/libriphone.yaml \
  -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt \
  -n pr_tera2_slim_960_200per_1e3 -o config.optimizer.lr=1.0e-3

# Evaluation
python3 run_downstream.py -m evaluate \
  -e result/downstream/pr_tera2_slim_960_200per_1e3/dev-best.ckpt
```

#### KS (Keyword Spotting)

```bash
# Training
python3 run_downstream.py -m train -d speech_commands -u tera2_local \
  -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt \
  -n ks_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4

# Evaluation
python3 run_downstream.py -m evaluate \
  -e result/downstream/ks_tera2_slim_960_200per_1e4/dev-best.ckpt
```

#### SID (Speaker Identification)

```bash
# Training
python3 run_downstream.py -m train -d voxceleb1 -u tera2_local \
  -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt \
  -n sid_tera2_slim_960_200per_1e2 -o config.optimizer.lr=1.0e-2

# Evaluation
python3 run_downstream.py -m evaluate \
  -e result/downstream/sid_tera2_slim_960_200per_1e2/dev-best.ckpt
```

#### ASV (Automatic Speaker Verification)

```bash
# Training
python3 run_downstream.py -m train -d sv_voxceleb1 -u tera2_local \
  -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt \
  -n asv_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4

# Evaluation
./downstream/sv_voxceleb1/test_expdir.sh \
  ./result/downstream/asv_tera2_slim_960_200per_1e4/ /media/andi611/1TBSSD/VoxCeleb1

# Alternative evaluation path
./downstream/sv_voxceleb1/test_expdir.sh \
  ./result/downstream/asv_tera2_slim_960_200per_1e4/ /work/a129195789/VoxCeleb1
```

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@ARTICLE{tera-ssl,
  author={Liu, Andy T. and Li, Shang-Wen and Lee, Hung-yi},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech}, 
  year={2021},
  volume={29},
  number={},
  pages={2351-2366},
  keywords={Task analysis;Predictive models;Acoustics;Speech processing;Training;Data models;Bit error rate;Self-supervised;pre-training;representation},
  doi={10.1109/TASLP.2021.3095662}}

```

## Contact

(mailto:liuandyt@gmail.com)
