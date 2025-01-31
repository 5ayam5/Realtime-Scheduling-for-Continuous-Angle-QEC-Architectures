if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path>"
    echo "Usage: $0 <path> <MAX_PROCESSES>"
    exit 1
fi

if [ "$#" -eq 2 ]; then
  MAX_PROCESSES=$2
else
  MAX_PROCESSES=6
fi

for dir in $1/*; do
    if [ -d "$dir" ]; then
      while [ $(jobs -pr | wc -l) -ge $MAX_PROCESSES ]; do
          sleep 1
      done
      echo "Postprocessing $dir"
      python3 postprocess/postprocess.py $dir &
    fi
done
wait
