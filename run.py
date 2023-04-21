import subprocess

subprocess.call(
    [
        "python3",
        "fairseq_cli/hydra_train.py",
        "--config-dir",
        "/work/a129195789/fairseq/examples/hubert/config/pretrain",
        "--config-name",
        "hubert_small_librispeech_debug",
        "task.data=/work/a129195789/fairseq/examples/hubert/simple_kmeans/data_dir/960",
        "task.label_dir=/work/a129195789/fairseq/examples/hubert/simple_kmeans/label_dir/960",
        "model.label_rate=100",
    ]
)