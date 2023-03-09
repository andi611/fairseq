for num in $(seq 0 $((3 - 1))); do
  cat ./data_dir/960/${num}.tsv
done > ./data_dir/960/train.tsv