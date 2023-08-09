# prepare train.tsv and valid.tsv for 960 hr
python3 examples/wav2vec/wav2vec_manifest.py /work/a129195789/LibriSpeech/ --dest /work/a129195789/manifest/960 --path-must-contain "train" --valid-percent 0
python3 examples/wav2vec/wav2vec_manifest.py /work/a129195789/LibriSpeech/ --dest /work/a129195789/manifest/dev-temp --path-must-contain "dev-clean" --valid-percent 0
mv /work/a129195789/manifest/960/dev-temp/train.tsv /work/a129195789/manifest/960/valid.tsv

# train:
# small
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/960 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_small_librispeech
# slim
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/960 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech
# slim %
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/960 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_slim_200per
# base %
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/960 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_base_467per
# large %
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/960 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining/exp --config-name wav2vec2_large_1559per

# prepare train.tsv and valid.tsv for 100 hr
python3 examples/wav2vec/wav2vec_manifest.py /work/a129195789/LibriSpeech/ --dest /work/a129195789/manifest/100 --path-must-contain "train-clean-100" --valid-percent 0
# train
# small
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/100 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_small_librispeech
# slim
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/100 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech

# prepare train.tsv and valid.tsv for 10 hr
python3 examples/wav2vec/sample_manifest.py
cp /work/a129195789/manifest/960/valid.tsv /work/a129195789/manifest/10/valid.tsv
# slim
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/10 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech

# prepare train.tsv and valid.tsv for 1 hr
vim examples/wav2vec/sample_manifest.py
python3 examples/wav2vec/sample_manifest.py
cp /work/a129195789/manifest/100/valid.tsv /work/a129195789/manifest/1/valid.tsv
# slim
python3 fairseq_cli/hydra_train.py task.data=/work/a129195789/manifest/1 --config-dir /work/a129195789/fairseq/examples/wav2vec/config/pretraining --config-name wav2vec2_slim_librispeech
