# pretraining:
## slim
python3 fairseq_cli/hydra_train.py --config-dir /path/to/fairseq/examples/tera/config/pretrain --config-name tera_slim_librispeech task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960

## slim %
python3 fairseq_cli/hydra_train.py --config-dir /path/to/fairseq/examples/tera/config/pretrain/exp --config-name tera_slim_200per_50flops_librispeech task.data=/path/to/fairseq/examples/hubert/simple_kmeans/data_dir/960 task.label_dir=/path/to/fairseq/examples/hubert/simple_kmeans/label_dir/960

# Downstream evaluation
## after pretraining, convert ckpt format with s3prl/s3prl
python3 upstream/tera2/convert.py result/tera2_slim_960_200per/checkpoints/checkpoint_best.pt --output_dir=result/tera2_slim_960_200per/converted_ckpts/
## or this path
python3 upstream/tera2/convert.py ../../fairseq/tera2_slim_960_200per/checkpoints/checkpoint_best.pt --output_dir=result/tera2_slim_960_200per/converted_ckpts/

## after convert, run downstream:
### asr
python3 run_downstream.py -m train -d asr -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt -n asr_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4
python3 run_downstream.py -m evaluate -t "test-clean" -e result/downstream/asr_tera2_slim_960_200per_1e4/dev-clean-best.ckpt

### pr
python3 run_downstream.py -m train -d ctc -c downstream/ctc/libriphone.yaml -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt -n pr_tera2_slim_960_200per_1e3 -o config.optimizer.lr=1.0e-3
python3 run_downstream.py -m evaluate -e result/downstream/pr_tera2_slim_960_200per_1e3/dev-best.ckpt

### ks
python3 run_downstream.py -m train -d speech_commands -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt -n ks_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4
python3 run_downstream.py -m evaluate -e result/downstream/ks_tera2_slim_960_200per_1e4/dev-best.ckpt

### sid
python3 run_downstream.py -m train -d voxceleb1 -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt -n sid_tera2_slim_960_200per_1e2 -o config.optimizer.lr=1.0e-2
python3 run_downstream.py -m evaluate -e result/downstream/sid_tera2_slim_960_200per_1e2/dev-best.ckpt

### asv
python3 run_downstream.py -m train -d sv_voxceleb1 -u tera2_local -k result/tera2_slim_960_200per/converted_ckpts/checkpoint_best.pt -n asv_tera2_slim_960_200per_1e4 -o config.optimizer.lr=1.0e-4
./downstream/sv_voxceleb1/test_expdir.sh ./result/downstream/asv_tera2_slim_960_200per_1e4/ /path/to/VoxCeleb1
