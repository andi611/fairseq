lab_dir="labels_iter1"
n_clusters=100
for x in $(seq 0 $(($n_clusters - 1))); do
  echo "$x 1"
done >> ./$lab_dir/dict.km.txt
