from collections import defaultdict
from os import listdir, path, makedirs
from math import floor
import argparse
from multiprocessing import Process
import matplotlib.pyplot as plt

COMPRESSIONS = [0]
COMPILERS = [
    "rescq_25",
    "rescq_50",
    "rescq_100",
    "rescq_200",
    "static",
    "AutoBraid",
]
ERROR_RATES = list(range(4, 5))
CODE_DISTANCES = [2 * i + 1 for i in range(3, 4)]
MST_FREQUENCY = [25 * (2**i) for i in range(4)]


def plot_histograms(
    input_folder: str,
    output_folder: str,
    compilerstr: str,
    compression: int,
    p: int,
    d: int,
):
    """
    function to plot histograms of Rz and CNOT execution times over all benchmarks
    """
    frequency = ""
    if compilerstr.find("_") != -1:
        compiler, frequency = compilerstr.split("_")
    else:
        compiler = compilerstr.lower()
    cnot_frequencies = defaultdict(float)
    rz_frequencies = defaultdict(float)
    total_cnot, total_rz = 0, 0
    for benchmark_class in listdir(path.join(input_folder, str(compression))):
        if not path.isdir(path.join(input_folder, str(compression), benchmark_class)):
            continue
        if frequency == "" and benchmark_class.find(f"{compiler}_{p}_{d}") == -1:
            continue
        if (
            frequency != ""
            and benchmark_class.find(f"{compiler}_{p}_{d}_{frequency}") == -1
        ):
            continue
        for benchmark in listdir(
            path.join(input_folder, str(compression), benchmark_class)
        ):
            if not path.isdir(
                path.join(input_folder, str(compression), benchmark_class, benchmark)
            ):
                continue
            with open(
                path.join(
                    input_folder,
                    str(compression),
                    benchmark_class,
                    benchmark,
                    "logs",
                    "CNOT",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    cnot_frequencies[floor(float(line.strip()))] += 1
                    total_cnot += 1
            with open(
                path.join(
                    input_folder,
                    str(compression),
                    benchmark_class,
                    benchmark,
                    "logs",
                    "Rz",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                lines = f.readlines()
                for line in lines:
                    rz_frequencies[floor(float(line.strip()))] += 1
                    total_rz += 1
    for key in cnot_frequencies.keys():
        cnot_frequencies[key] *= 100 / total_cnot
    for key in rz_frequencies.keys():
        rz_frequencies[key] *= 100 / total_rz
    for key in list(cnot_frequencies.keys()):
        if cnot_frequencies[key] < 0.1 or key > 15:
            del cnot_frequencies[key]
    for key in list(rz_frequencies.keys()):
        if rz_frequencies[key] < 0.1 or key > 20:
            del rz_frequencies[key]
    cnot_mean = sum(key * value for key, value in cnot_frequencies.items()) / 100
    rz_mean = sum(key * value for key, value in rz_frequencies.items()) / 100
    cnot_median, rz_median = -1, -1
    cnot_90, rz_90 = -1, -1
    cnot_sum, rz_sum = 0, 0
    for key in sorted(cnot_frequencies.keys()):
        cnot_sum += cnot_frequencies[key]
        if cnot_sum >= 50 and cnot_median == -1:
            cnot_median = key
        if cnot_sum >= 90 and cnot_90 == -1:
            cnot_90 = key
    for key in sorted(rz_frequencies.keys()):
        rz_sum += rz_frequencies[key]
        if rz_sum >= 50 and rz_median == -1:
            rz_median = key
        if rz_sum >= 90 and rz_90 == -1:
            rz_90 = key

    plt.rcParams["figure.figsize"] = (10, 3.75)
    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"{compiler} ($p = 10^{{-{p}}}, d = {d}$)")

    axs[0].bar(cnot_frequencies.keys(), cnot_frequencies.values(), color="orange")
    axs[0].axvline(x=cnot_mean, color="b", linestyle="-", label="Mean")
    axs[0].axvline(x=cnot_median, color="g", linestyle="--", label="Median")
    axs[0].axvline(x=cnot_90, color="black", linestyle="-.", label="90th percentile")
    axs[0].set_title("CNOT")
    axs[0].set_xlabel("Number of lattice surgery cycles to completion")
    axs[0].set_ylabel("Percentage of gates")
    axs[0].set_ylim(0, 75)
    axs[0].set_xlim(0, 15)
    axs[0].legend()

    axs[1].bar(rz_frequencies.keys(), rz_frequencies.values(), color="orange")
    axs[1].axvline(x=rz_mean, color="b", linestyle="-", label="Mean")
    axs[1].axvline(x=rz_median, color="g", linestyle="--", label="Median")
    axs[1].axvline(x=rz_90, color="black", linestyle="-.", label="90th percentile")
    axs[1].set_title("$Rz$")
    axs[1].set_xlabel("Number of lattice surgery cycles to completion")
    axs[1].set_ylim(0, 75)
    axs[1].set_xlim(0, 20)
    axs[1].legend()

    plt.tight_layout()
    if not path.exists(path.join(output_folder, f"sensitivity_{compression}")):
        makedirs(path.join(output_folder, f"sensitivity_{compression}"))
    plt.savefig(
        path.join(
            output_folder, f"sensitivity_{compression}", f"{compilerstr}_{p}_{d}.svg"
        )
    )
    plt.close()


def main(input_folder: str, output_folder: str):
    """
    main function for this file
    """
    for compression in COMPRESSIONS:
        for compiler in COMPILERS:
            procs = []
            for error_rate in ERROR_RATES:
                for code_distance in CODE_DISTANCES:
                    print(
                        f"Processing {compression}_{compiler}_{error_rate}_{code_distance}",
                        flush=True,
                    )
                    proc = Process(
                        target=plot_histograms,
                        args=(
                            input_folder,
                            output_folder,
                            compiler,
                            compression,
                            error_rate,
                            code_distance,
                        ),
                    )
                    proc.start()
                    procs.append(proc)
            for proc in procs:
                proc.join()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("input_directory", type=str)
    args.add_argument("output_directory", type=str)
    args = args.parse_args()
    main(args.input_directory, args.output_directory)
