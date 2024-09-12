# TERA - A Fairseq Implementation
## Speech Foundation Model Pre-training and Evaluation

This directory contains modified [Fairseq](https://github.com/facebookresearch/fairseq) code for pre-training TERA, a self-supervised speech foundation model from the paper [TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](https://arxiv.org/abs/2007.06028). While upstream pre-training uses the Fairseq toolkit, the downstream evaluation uses the [S3PRL](https://github.com/s3prl/s3prl) toolkit.

This README also lists the upstream pre-training & downstream evaluation commands used in our recent paper [Efficient Training of Self-Supervised Speech Foundation Models\\on a Compute Budget]() for TERA.

- For HuBERT pre-training, please see the instructions [here](https://github.com/andi611/fairseq/tree/master/examples/hubert) and example commands [here](https://github.com/andi611/fairseq/blob/master/examples/hubert/commands.sh).

- For wav2vec 2.0 pre-training, please see the instructions [here](https://github.com/andi611/fairseq/tree/master/examples/wav2vec) and example commands [here](https://github.com/andi611/fairseq/blob/master/examples/wav2vec/commands.sh).

The downstream evaluation process of these models is very similar to TERA, and one can also refer to the official [downstream doc](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md) in S3PRL.

## Upstream Pre-training

This version of TERA is modified from the HuBERT example ([HuBERT in Fairseq](https://github.com/andi611/fairseq/tree/master/examples/hubert)) with minimal modification of code. As a result, the `task.label_dir` parameter in the following commands is not actually used in the TERA implementation but is required due to the HuBERT-based code structure. Please follow the instructions in the [HuBERT example directory](https://github.com/andi611/fairseq/tree/master/examples/hubert) to obtain this.

### Installing Fairseq

This implementation relies on [Fairseq](https://github.com/facebookresearch/fairseq). To install it, run the following commands:
```bash
git clone https://github.com/andi611/fairseq.git
cd fairseq
pip install --editable ./
```
Read [here](https://github.com/andi611/fairseq/tree/master) for more details.

### _Slim_ Model

To pre-train the TERA _Slim_ model on LibriSpeech:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq/examples/tera/config/pretrain/ \
  --config-name tera_slim_librispeech.yaml \
  task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

### _Slim_ Model with Different % Sizes

To pre-train the TERA _Slim_ model with different % sizes on LibriSpeech:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq/examples/tera/config/pretrain/exp \
  --config-name tera_slim_200per_50flops_librispeech \
  task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

## Downstream Evaluation

### Installing S3PRL

Before proceeding with downstream evaluation, you must install the S3PRL toolkit from a specific branch. Use the following command:

```bash
git clone -b tera2 https://github.com/s3prl/s3prl.git
cd s3prl
pip install -e .
```

This will install the S3PRL toolkit from the `tera2` branch, which is required for the downstream evaluation tasks in this project.

### Converting Checkpoint Format

After the above pretraining, first convert the checkpoint format using S3PRL:

```bash
python3 upstream/tera2/convert.py \
  result/tera2_slim_960_200per/checkpoints/checkpoint_best.pt \
  --output_dir=result/tera2_slim_960_200per/converted_ckpts/
```

### Running Downstream Tasks
Here we list downstream evaluation commands using S3PRL as examples.

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
  ./result/downstream/asv_tera2_slim_960_200per_1e4/ /path/to/1TBSSD/VoxCeleb1
```

## Citation

If you use this code for your research, please cite our papers:

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

@inproceedings{superb,
  author={Shu-Wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-Yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```

## Contact

(mailto:liuandyt@gmail.com)
(mailto:f07942089@ntu.edu.tw)
(mailto:f07921092@ntu.edu.tw)
