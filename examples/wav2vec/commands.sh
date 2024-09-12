# Prepare training data
## prepare train.tsv and valid.tsv for 960 hr
python3 examples/wav2vec/wav2vec_manifest.py /path/to/LibriSpeech/ --dest /path/to/manifest/960 --path-must-contain "train" --valid-percent 0
python3 examples/wav2vec/wav2vec_manifest.py /path/to/LibriSpeech/ --dest /path/to/manifest/dev-temp --path-must-contain "dev-clean" --valid-percent 0
mv /path/to/manifest/960/dev-temp/train.tsv /path/to/manifest/960/valid.tsv

# Start training:
## small
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/960 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_small_librispeech
## slim
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/960 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech
## slim %
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/960 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_slim_200per
## base %
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/960 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_base_467per
## large %
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/960 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_large_1559per

## prepare train.tsv and valid.tsv for 100 hr
python3 examples/wav2vec/wav2vec_manifest.py /path/to/LibriSpeech/ --dest /path/to/manifest/100 --path-must-contain "train-clean-100" --valid-percent 0

# Start training:
## small
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/100 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_small_librispeech
## slim
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/100 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech

## prepare train.tsv and valid.tsv for 10 hr
python3 examples/wav2vec/sample_manifest.py
cp /path/to/manifest/960/valid.tsv /path/to/manifest/10/valid.tsv

# Start training:
## slim
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/10 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech

## prepare train.tsv and valid.tsv for 1 hr
vim examples/wav2vec/sample_manifest.py
python3 examples/wav2vec/sample_manifest.py
cp /path/to/manifest/100/valid.tsv /path/to/manifest/1/valid.tsv

# Start training:
## slim
python3 fairseq_cli/hydra_train.py task.data=/path/to/manifest/1 --config-dir /path/to/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech
