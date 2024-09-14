# TERA - A Fairseq Implementation

This directory contains the upstream pre-training & downstream evaluation commands used in our recent paper:
[Efficient Training of Self-Supervised Speech Foundation Models on a Compute Budget](pending-arxiv-link).

While upstream pre-training uses the [Fairseq](https://github.com/facebookresearch/fairseq) toolkit,
the downstream evaluation uses the [S3PRL](https://github.com/s3prl/s3prl) toolkit.

## Background

The three self-supervised learning (SSL) objectives (HuBERT, wav2vec 2.0, TERA) can be standardized with identical model components and trained using the same toolkit.
This standardization allows us to use the same model architecture across different SSL objectives, a comparison not previously examined in the literature.
We minimize potential confounding factors that could influence final performance by pre-training various SSL objectives with consistent building blocks.
In our paper, we construct all models using consistent components, including a convolutional encoder, Transformer encoder blocks, and a projection layer.
We implement and train these models using the Fairseq toolkit.

## Speech Foundation Model Pre-training

- For HuBERT pre-training, please see the [instructions](https://github.com/andi611/fairseq/tree/master/examples/hubert) and [example commands](https://github.com/andi611/fairseq/blob/master/examples/hubert/commands.sh). Note that [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) was initially proposed with the Fairseq implementation; no modification is needed.

- For wav2vec 2.0 pre-training, please see the [instructions](https://github.com/andi611/fairseq/tree/master/examples/wav2vec) and [example commands](https://github.com/andi611/fairseq/blob/master/examples/wav2vec/commands.sh). Note that [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) was initially proposed with the Fairseq implementation; no modification is needed.

- For TERA pre-training, this directory contains the modified Fairseq pre-training code. TERA is a self-supervised speech foundation model from the paper [TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](https://arxiv.org/abs/2007.06028), initially not implemented with Fairseq.

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1oUDZEdSjGATd-tJf_7Re-67HnBeoZ7NM?usp=sharing).

The downstream evaluation processes of these models are very similar to each other; one can also refer to the official [downstream documentation](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md) on S3PRL for a more detailed guide (data setup, preprocessing, config usage, etc).

## TERA Upstream Pre-training

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

To pre-train the TERA 100% _Slim_ model on LibriSpeech:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq/examples/tera/config/pretrain/ \
  --config-name tera_slim_librispeech.yaml \
  task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

### _Slim_ Model with Different % Sizes

To pre-train the TERA _Slim_ model with different % sizes on LibriSpeech, 200% for example:

```bash
python3 fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq/examples/tera/config/pretrain/exp \
  --config-name tera_slim_200per_librispeech \
  task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 \
  task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960
```

## TERA Downstream Evaluation

The downstream evaluation of TERA is on the `tera2` [branch](https://github.com/s3prl/s3prl/tree/tera2) of S3PRL. The TERA upstream is integrated into the S3PRL toolkit [here](https://github.com/s3prl/s3prl/tree/tera2/s3prl/upstream/tera2).

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
@ARTICLE{tera,
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

- Corresponding author: [Andy T. Liu](mailto:liuandyt@gmail.com)
- FLOPS related inquiries can also go to: [Yi-Cheng Lin](mailto:r12942075@ntu.edu.tw)
- Implementation related inquiries can also go to: [Haibin Wu](mailto:f07921092@ntu.edu.tw)
