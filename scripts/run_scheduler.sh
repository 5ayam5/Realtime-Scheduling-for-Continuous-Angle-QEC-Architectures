if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    echo "Usage: $0 <scheduler> <subdirectory in config/>"
    echo "Usage: $0 <scheduler> <subdirectory in config/> <MAX_PROCESSES>"
    exit 1
fi

scheduler=$1
subdir=$2
if [ "$#" -eq 3 ]; then
  MAX_PROCESSES=$3
else
  MAX_PROCESSES=6
fi

for dir in config/$subdir/*_$scheduler; do
    if [ -d "$dir" ]; then
        for config in $dir/*; do
            if [ -f "$config" ]; then
                while [ $(jobs -pr | wc -l) -ge $MAX_PROCESSES ]; do
                    sleep 1
                done
                echo "Running $config..."
                ./build/sim $config &
                # TODO: remove the below comments if I ever switch back to non-forked execution
                # echo "Done."
                # echo
            fi
        done
    fi
done
wait
