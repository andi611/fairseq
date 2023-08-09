# Install
sudo apt install python
sudo apt install python3-pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -U pip3

# Iter 1
# Feature extraction
python3 dump_mfcc_feature.py ./data_dir/100 train 1 0 ./feat_dir/100
python3 dump_mfcc_feature.py ./data_dir/960 valid 1 0 ./feat_dir/960

# K-means clustering
python3 learn_kmeans.py ./feat_dir/100 train 1 ./km_dir/960/km_train 100 --percent 1.0
python3 learn_kmeans.py ./feat_dir/960 train 1 ./km_dir/960/km_train 100 --percent 1.0

# K-mean application
python3 dump_km_label.py ./feat_dir/100 train ./km_dir/960/km_train 1 0 ./label_dir/100
python3 dump_km_label.py ./feat_dir/960 valid ./km_dir/960/km_train 1 0 ./label_dir/960


# train iter 1
# small
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960 task.labels='["km"]' model.label_rate=100
# slim
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_slim_librispeech task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960 task.labels='["km"]' model.label_rate=100


# Iter 2
# Feature extraction on hubert
python3 dump_hubert_feature.py ./data_dir/960 train /work/a129195789/fairseq/hubert_small_960_iter1/checkpoints/checkpoint_best.pt 2 1 0 ./feat_dir/960_iter1
python3 dump_hubert_feature.py ./data_dir/960 valid /work/a129195789/fairseq/hubert_small_960_iter1/checkpoints/checkpoint_best.pt 2 1 0 ./feat_dir/960_iter1
python3 dump_hubert_feature.py ./data_dir/100 train /work/a129195789/fairseq/hubert_small_iter1/checkpoints/checkpoint_last.pt 2 1 0 ./feat_dir/100_iter1
python3 dump_hubert_feature.py ./data_dir/100 valid /work/a129195789/fairseq/hubert_small_iter1/checkpoints/checkpoint_last.pt 2 1 0 ./feat_dir/100_iter1
# slim
python3 dump_hubert_feature.py ./data_dir/960 train /work/a129195789/fairseq/hubert_slim_960_iter1/checkpoints/checkpoint_best.pt 6 1 0 ./feat_dir/960_slim_iter1
python3 dump_hubert_feature.py ./data_dir/960 valid /work/a129195789/fairseq/hubert_slim_960_iter1/checkpoints/checkpoint_best.pt 6 1 0 ./feat_dir/960_slim_iter1


# K-means clustering
python3 learn_kmeans.py ./feat_dir/960_iter1 train 1 ./km_dir/960/km_train_iter1 500 --percent 0.1
python3 learn_kmeans.py ./feat_dir/100_iter1 train 1 ./km_dir/100/km_train_iter1 500 --percent 1.0
# slim
python3 learn_kmeans.py ./feat_dir/960_slim_iter1 train 1 ./km_dir/960/km_train_slim_iter1 500 --percent 0.1

# K-mean application
python3 dump_km_label.py ./feat_dir/960_iter1 train ./km_dir/960/km_train_iter1 1 0 ./label_dir/960_iter1
python3 dump_km_label.py ./feat_dir/960_iter1 valid ./km_dir/960/km_train_iter1 1 0 ./label_dir/960_iter1
python3 dump_km_label.py ./feat_dir/100_iter1 train ./km_dir/100/km_train_iter1 1 0 ./label_dir/100_iter1
python3 dump_km_label.py ./feat_dir/100_iter1 valid ./km_dir/100/km_train_iter1 1 0 ./label_dir/100_iter1
# slim
python3 dump_km_label.py ./feat_dir/960_slim_iter1 train ./km_dir/960/km_train_slim_iter1 1 0 ./label_dir/960_slim_iter1
python3 dump_km_label.py ./feat_dir/960_slim_iter1 valid ./km_dir/960/km_train_slim_iter1 1 0 ./label_dir/960_slim_iter1

# train iter 2
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_iter1 task.labels='["km"]' model.label_rate=50
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/100 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/100_iter1 task.labels='["km"]' model.label_rate=50
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_slim_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_slim_iter1 task.labels='["km"]' model.label_rate=50

# train slim %
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain/exp --config-name hubert_large_1582per task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_slim_iter1 task.labels='["km"]' model.label_rate=50