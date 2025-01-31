process() {
  compression=$1
  compression_frac=$(echo "scale=6; $compression / 100" | bc)
  benchmark=$2
  scheduler=$3

  if [ ! -d "configs" ]; then
    mkdir "configs"
  fi

  name="$benchmark"'_'"$scheduler"
  if [ ! -d "configs/""$compression"'_'"$name" ]; then
    mkdir "configs/""$compression"'_'"$name"
  fi
  if [ ! -f "$name"'.cfg' ]; then
    echo "Error: $name.cfg not found"
    return 1
  fi

  if [ "$#" -eq 3 ]; then
    p=4
    prob=$(echo "scale=6; 10^-$p" | bc)
    for d in $(seq 3 2 11); do
      if [ "$scheduler" = "rescq" ]; then
        for freq in 25 50 100 200; do
          variant="$name"'_'"$p"'_'"$d"'_'"$freq"
          cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/mst_computation_frequency/s/=.*/= '"$freq"'.0/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
        done
      else
        variant="$name"'_'"$p"'_'"$d"
        cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
      fi
    done

    d=7
    for p in $(seq 3 1 6); do
      prob=$(echo "scale=6; 10^-$p" | bc)
      if [ "$scheduler" = "rescq" ]; then
        for freq in 25 50 100 200; do
          variant="$name"'_'"$p"'_'"$d"'_'"$freq"
          cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/mst_computation_frequency/s/=.*/= '"$freq"'.0/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
        done
      else
        variant="$name"'_'"$p"'_'"$d"
        cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
      fi
    done
  elif [ "$#" -eq 5 ]; then
    p=$4
    prob=$(echo "scale=6; 10^-$p" | bc)
    d=$5
    if [ "$scheduler" = "rescq" ]; then
      for freq in 25 50 100 200; do
        variant="$name"'_'"$p"'_'"$d"'_'"$freq"
        cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/mst_computation_frequency/s/=.*/= '"$freq"'.0/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
      done
    else
      variant="$name"'_'"$p"'_'"$d"
      cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
    fi
  elif [ "$#" -eq 6 ]; then
    p=$4
    prob=$(echo "scale=6; 10^-$p" | bc)
    d=$5
    freq=$6
    if [ "$scheduler" = "rescq" ]; then
      variant="$name"'_'"$p"'_'"$d"'_'"$freq"
      cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/mst_computation_frequency/s/=.*/= '"$freq"'.0/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
    else
      variant="$name"'_'"$p"'_'"$d"
      cat "$name"'.cfg' | sed '/compression_factor/s/=.*/= '"$compression_frac"'/' | sed '/physical_qubit_error_rate/s/=.*/= '"$prob"'/' | sed '/code_distance/s/=.*/= '"$d"'/' | sed '/output_dir/s/\/.*"/\/'"$compression"'\/'"$variant"'"/' > "configs/""$compression"'_'"$name"'/'"$variant"'.cfg'
    fi
  else
    echo "Error: Invalid number of arguments"
    return 1
  fi

  return 0
}

single_compression() {
  compression=$1
  for benchmark in "large" "medium" "supermarq"; do
    for scheduler in "static" "rescq" "autobraid"; do
      process "$compression" "$benchmark" "$scheduler"
    done
  done
}

if [ "$#" -ne 3 ] && [ "$#" -ne 2 ] && [ "$#" -ne 1 ]; then
  echo "Usage: $0 <compression> <benchmark> <scheduler>"
  echo "Usage: $0 <compression>"
  echo "Usage: $0 all"
  echo "Usage: $0 <compression> dp"
  echo "Usage: $0 <compression> dpf"
  exit 1
fi

if [ "$#" -eq 3 ]; then
  process "$1" "$2" "$3"
elif [ "$#" -eq 1 ]; then
  compression=$1
  if [ "$compression" = "all" ]; then
    for compression in 0 25 50 75 100; do
      single_compression "$compression"
    done
  else
    single_compression "$compression"
  fi
else
  if [ "$2" = "dp" ]; then
    compression=$1
    for benchmark in "large" "medium" "supermarq"; do
      for scheduler in "static" "rescq" "autobraid"; do
        process "$compression" "$benchmark" "$scheduler" 4 7
      done
    done
  elif [ "$2" = "dpf" ]; then
    compression=$1
    for benchmark in "large" "medium" "supermarq"; do
      for scheduler in "static" "rescq" "autobraid"; do
        process "$compression" "$benchmark" "$scheduler" 4 7 25
      done
    done
  else
    echo "Error: Invalid argument"
    exit 1
  fi
fi

