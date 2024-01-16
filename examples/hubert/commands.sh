# Install
sudo apt install python
sudo apt install python3-pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -U pip3

# Prepare Iter 1
## Feature extraction
python3 dump_mfcc_feature.py ./data_dir/100 train 1 0 ./feat_dir/100
python3 dump_mfcc_feature.py ./data_dir/960 valid 1 0 ./feat_dir/960

## K-means clustering
python3 learn_kmeans.py ./feat_dir/100 train 1 ./km_dir/960/km_train 100 --percent 1.0
python3 learn_kmeans.py ./feat_dir/960 train 1 ./km_dir/960/km_train 100 --percent 1.0

## K-mean application
python3 dump_km_label.py ./feat_dir/100 train ./km_dir/960/km_train 1 0 ./label_dir/100
python3 dump_km_label.py ./feat_dir/960 valid ./km_dir/960/km_train 1 0 ./label_dir/960


# Train iter 1
## small
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960 task.labels='["km"]' model.label_rate=100
## slim
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_slim_librispeech task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960 task.labels='["km"]' model.label_rate=100


# Prepare iter 2
## Feature extraction on hubert small
python3 dump_hubert_feature.py ./data_dir/960 train /work/a129195789/fairseq/hubert_small_960_iter1/checkpoints/checkpoint_best.pt 2 1 0 ./feat_dir/960_iter1
python3 dump_hubert_feature.py ./data_dir/960 valid /work/a129195789/fairseq/hubert_small_960_iter1/checkpoints/checkpoint_best.pt 2 1 0 ./feat_dir/960_iter1
python3 dump_hubert_feature.py ./data_dir/100 train /work/a129195789/fairseq/hubert_small_iter1/checkpoints/checkpoint_last.pt 2 1 0 ./feat_dir/100_iter1
python3 dump_hubert_feature.py ./data_dir/100 valid /work/a129195789/fairseq/hubert_small_iter1/checkpoints/checkpoint_last.pt 2 1 0 ./feat_dir/100_iter1
## Feature extraction on hubert slim
python3 dump_hubert_feature.py ./data_dir/960 train /work/a129195789/fairseq/hubert_slim_960_iter1/checkpoints/checkpoint_best.pt 6 1 0 ./feat_dir/960_slim_iter1
python3 dump_hubert_feature.py ./data_dir/960 valid /work/a129195789/fairseq/hubert_slim_960_iter1/checkpoints/checkpoint_best.pt 6 1 0 ./feat_dir/960_slim_iter1

## K-means clustering on hubert small
python3 learn_kmeans.py ./feat_dir/960_iter1 train 1 ./km_dir/960/km_train_iter1 500 --percent 0.1
python3 learn_kmeans.py ./feat_dir/100_iter1 train 1 ./km_dir/100/km_train_iter1 500 --percent 1.0
## K-means clustering on hubert slim
python3 learn_kmeans.py ./feat_dir/960_slim_iter1 train 1 ./km_dir/960/km_train_slim_iter1 500 --percent 0.1

## K-mean application on hubert small
python3 dump_km_label.py ./feat_dir/960_iter1 train ./km_dir/960/km_train_iter1 1 0 ./label_dir/960_iter1
python3 dump_km_label.py ./feat_dir/960_iter1 valid ./km_dir/960/km_train_iter1 1 0 ./label_dir/960_iter1
python3 dump_km_label.py ./feat_dir/100_iter1 train ./km_dir/100/km_train_iter1 1 0 ./label_dir/100_iter1
python3 dump_km_label.py ./feat_dir/100_iter1 valid ./km_dir/100/km_train_iter1 1 0 ./label_dir/100_iter1
## K-mean application on hubert slim
python3 dump_km_label.py ./feat_dir/960_slim_iter1 train ./km_dir/960/km_train_slim_iter1 1 0 ./label_dir/960_slim_iter1
python3 dump_km_label.py ./feat_dir/960_slim_iter1 valid ./km_dir/960/km_train_slim_iter1 1 0 ./label_dir/960_slim_iter1

# Train iter 2
## small
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_iter1 task.labels='["km"]' model.label_rate=50
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_small_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/100 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/100_iter1 task.labels='["km"]' model.label_rate=50
## slim
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain --config-name hubert_slim_librispeech_iter2 task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_slim_iter1 task.labels='["km"]' model.label_rate=50

## train slim % flops curve
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain/exp --config-name hubert_large_1582per task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_slim_iter1 task.labels='["km"]' model.label_rate=50
## train slim % flops curve at 50% flops
python3 fairseq_cli/hydra_train.py --config-dir /work/a129195789/fairseq/examples/hubert/config/pretrain/exp2_half_flops --config-name hubert_slim_50per task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960_slim_iter1 task.labels='["km"]' model.label_rate=50

# Downstream evaluation
## after pretraing, convert ckpt format with s3prl/s3prl
python3 upstream/hubert/convert.py result/hubert_slim_960_200per_50flops/checkpoints/checkpoint_best.pt --output_dir=result/hubert_slim_960_200per_50flops/converted_ckpts/
python3 upstream/hubert/convert.py ../../fairseq/hubert_slim_960_100per_50flops/checkpoints/checkpoint_best.pt --output_dir=result/hubert_slim_960_100per_50flops/converted_ckpts/

## after convert, run downstream:
### asr
python3 run_downstream.py -m train -d asr -u hubert_local -k result/hubert_large_960_1582per_50flops/converted_ckpts/checkpoint_best.pt -n asr_hubert_large_960_1582per_50flops_1e4 -o config.optimizer.lr=1.0e-4
python3 run_downstream.py -m evaluate -t "test-clean" -e result/downstream/asr_hubert_large_960_1582per_50flops_1e4/dev-clean-best.ckpt

### pr
python3 run_downstream.py -m train -d ctc -c downstream/ctc/libriphone.yaml -u hubert_local -k result/hubert_slim_960_200per/converted_ckpts/checkpoint_best.pt -n pr_hubert_slim_960_200per_1e3 -o config.optimizer.lr=1.0e-3
python3 run_downstream.py -m evaluate -e result/downstream/pr_hubert_slim_960_200per_1e3/dev-best.ckpt

### asv
python3 run_downstream.py -m train -d sv_voxceleb1 -u hubert_local -k result/hubert_slim_960_100per_50flops/converted_ckpts/checkpoint_best.pt -n asv_hubert_slim_960_100per_50flops_1e4 -o config.optimizer.lr=1.0e-4
./downstream/sv_voxceleb1/test_expdir.sh ./result/downstream/asv_hubert_large_960_1582per_50flops_1e4/ /media/andi611/1TBSSD/VoxCeleb1