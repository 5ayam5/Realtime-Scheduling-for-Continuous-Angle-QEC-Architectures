if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <max_processes>"
  exit 1
fi

cwd=$(pwd)
echo "Current working directory: $cwd"
MAX_PROCESSES=$1
echo "Max processes: $MAX_PROCESSES"

cd "config"
if [ $? -ne 0 ]; then
  echo "Failed to change directory to config"
  echo "Make sure you are running the script from the root of the repository"
  exit 1
fi

if [ ! -f "../scripts/gen_configs.sh" ]; then
  echo "Failed to find gen_configs.sh"
  echo "Double check if the repository is cloned correctly"
  exit 1
fi
../scripts/gen_configs.sh 0
../scripts/gen_configs.sh 50 dp
../scripts/gen_configs.sh 25 dpf
../scripts/gen_configs.sh 75 dpf
../scripts/gen_configs.sh 100 dpf
echo "Generated configs"

cd ".."
for scheduler in "static" "rescq" "autobraid"; do
  echo "Running $scheduler"
  ./scripts/run_scheduler.sh $scheduler configs $MAX_PROCESSES
done

echo "Postprocessing outputs"
for dir in outputs/*; do
  if [ -d $dir ]; then
    ./scripts/run_postprocess.sh $dir $MAX_PROCESSES
  fi
done
