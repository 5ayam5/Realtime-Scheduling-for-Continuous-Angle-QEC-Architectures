# RESCQ: Realtime Scheduling for Continuous Angle QEC Architectures

Simulator for the paper "RESCQ: Realtime Scheduling for Continuous Angle Quantum Error Correction Architectures".

## Requirements
The compilation uses [cmake](https://cmake.org). The [Boost](https://www.boost.org) library is also required for compiling the simulator. You can install it with the following command for Debian-based systems:
```bash
sudo apt-get install libboost-all-dev
```
For MacOS, you can install it with the following command (assuming you have homebrew installed):
```bash
brew install boost
```

### Postprocessing
The Python packages required to run the postprocessing scripts in `postprocess` directory are listed in `postprocess/requirements.txt`. Please use [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) to install these packages in a fresh Python environment.

## Compilation
To compile the simulator, run the following command:
```bash
mkdir -p build
cd build
cmake ..
make -j4
cd ..
```
The executable will be compiled as `sim` in the `build` directory.

## Usage
To run the simulator, use the following command from the base directory:
```bash
./build/sim <config_file>
```
The supported parameters in the `<config_file>` with their default values are described in [Configuration File](#configuration-file).

### Postprocessing
To run the postprocessing script on the logs generated by `sim`, run the following command:
```bash
python postprocess/postprocessing.py <output_dir>
```
The `<output_dir>` is the same directory specified in the `<config_file>` when running the simulator.

### Installation Script and Basic Testing
To check if the simulator works correcly post-installation, a basic test script `basic_test.sh` is provided in the `scripts` directory. The script assumes the executable `sim` has been generated in the `build` directory. It runs the simulator with the `basic_test.cfg` configuration and performs post-processing on the output directory (`outputs/basic_test`). To run the test script, simply run:
```bash
./scripts/basic_test.sh
```
Note that the above command should be run from the base directory of the installation.

## Generating the Plots from the Paper
To generate the logs that are used to generate the plots used in the paper, simply run:
```bash
./scripts/run_all.sh <MAX_NUMBER_OF_THREADS_TO_USE>
```
> [!IMPORTANT]
> The above script takes about an hour on 16 cores and the logs generated requires about 1-2 GB of disk space.

## Configuration File
The configuration file is used to pass the parameters to the simulator. The configuration file has the following format:
```
<parameter_name> = <parameter_value>
```

### Supported Parameters and Descriptions
- `input_dir`: The input directory containing the circuits to be executed. (Default: `input_dir`)
- `input_file`: Regular expression for the circuits to be executed in the input directory. (Default: `gate_.*`)
- `output_dir`: The output file where the simulation results will be stored. (Default: `output_dir`)
- `code_distance`: The code distance of the surface code.
- `physical_qubit_error_rate`: The error rate of the physical qubits.
- `debug`: If set to `1`, the simulator will print debug information. (Default: `0`)
- `number_of_runs`: The number of times to run each circuit (on different seeds). (Default: `100`)
- `scheduler`: The scheduling algorithm to use. Supported values are `rescq`, `static` and `autobraid`. (Default: `rescq`)
- `compression_factor`: The compression factor for the surface code grid. Supported range is between `0-1`. (Default: `0`)
- `mst_computation_frequency`: The frequency (in cycles) of the MST computation. (Default: `100`)
