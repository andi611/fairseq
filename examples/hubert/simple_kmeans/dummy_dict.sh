n_clusters = 100
for x in $(seq 0 $(($n_clusters - 1))); do
  echo "$x 1"
done >> ./labels/dict.km.txt