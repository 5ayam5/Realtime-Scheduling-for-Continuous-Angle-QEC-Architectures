from collections import defaultdict
from math import inf
from os import path, makedirs
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


COLOURS = {
    "rescq": cm.get_cmap("viridis")(0),
    "static": cm.get_cmap("viridis")(0.4),
    "autobraid": cm.get_cmap("viridis")(0.8),
}
LINE_STYLES = {"rescq": "-", "static": "--", "autobraid": "-."}
COMPRESSIONS = [50]
COMPILERS = [
    "rescq_25",
    "rescq_50",
    "rescq_100",
    "rescq_200",
    "static",
    "autobraid",
]
COMPILER_MAP = {"static": "greedy", "autobraid": "AutoBraid", "rescq": r"$rescq^*$"}
ERROR_RATES = list(range(4, 5))
CODE_DISTANCES = [2 * i + 1 for i in range(3, 4)]
MST_FREQUENCY = [25 * (2**i) for i in range(4)]

BENCHMARK_CLASSES = {
    "medium": ["gates_gcm_n13", "gates_dnn_n16", "gates_qft_n18", "gates_wstate_n27"],
    "large": [
        "gates_ising_n34",
        "gates_ising_n42",
        "gates_ising_n66",
        "gates_ising_n98",
        "gates_ising_n420",
        "gates_multiplier_n45",
        "gates_multiplier_n75",
        "gates_qft_n29",
        "gates_qft_n63",
        "gates_qft_n160",
        "gates_qugan_n39",
        "gates_qugan_n71",
        "gates_qugan_n111",
    ],
    # "long": {"gates_qft_n320", "gates_vqe_uccsd_n28"},
    "supermarq": [
        "gates_QAOAFermionicSwapProxy_15",
        "gates_QAOAVanillaProxy_15",
        "gates_VQEProxy_13",
        "gates_HamiltonianSimulation_25",
        "gates_HamiltonianSimulation_50",
        "gates_HamiltonianSimulation_75",
    ],
}


def benchmark_short_name(benchmark: str):
    """
    function to get the short name of the benchmark
    """
    benchmark = benchmark[6:]
    if benchmark.find("_n") == -1:
        benchmark = benchmark.replace("_", "_n")
    if benchmark.find("Proxy") != -1:
        benchmark = benchmark.replace("Proxy", "")
        if benchmark.find("QAOA") != -1:
            benchmark = benchmark.replace("Vanilla", "")
            benchmark = benchmark.replace("FermionicSwap", "FermSwp")
        return benchmark
    if benchmark.find("HamiltonianSimulation") != -1:
        return benchmark.replace("HamiltonianSimulation", "HamSim")
    return benchmark


def generate_final_plots(
    input_directory: str,
    output_directory: str,
    compression: int,
    error_rate: int,
    code_distance: int,
    speedup_file: str,
):
    """
    function to generate the final execution plots for the paper
    """
    plt.rcParams["figure.figsize"] = (27.5, 6)
    plt.rcParams.update({"font.size": 18})
    # plt.tick_params(axis='x', which='both', labelsize=10)
    benchmark_dict = defaultdict(lambda: defaultdict(lambda: (inf, inf, -inf)))

    for benchmark_class, benchmarks in BENCHMARK_CLASSES.items():
        for benchmark in benchmarks:
            for compilerstr in COMPILERS:
                compiler, frequency = "", ""
                if compilerstr.find("_") != -1:
                    compiler, frequency = compilerstr.split("_")
                else:
                    compiler = compilerstr
                if frequency != "":
                    current_directory = path.join(
                        input_directory,
                        str(compression),
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}_{frequency}",
                        benchmark,
                    )
                else:
                    current_directory = path.join(
                        input_directory,
                        str(compression),
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}",
                        benchmark,
                    )
                with open(
                    path.join(current_directory, "log"), "r", encoding="utf-8"
                ) as f:
                    lines = f.readlines()
                times = []
                for line in lines:
                    if line.startswith("Done in"):
                        times.append(float(line.split()[2]))
                if benchmark_dict[benchmark][compiler][0] > sum(times) / len(times):
                    benchmark_dict[benchmark][compiler] = (
                        sum(times) / len(times),
                        min(times),
                        max(times),
                    )

    n_bars = 3
    bar_width = 0.5 / n_bars

    labels = {}
    geomeans = defaultdict(lambda: 1.0)
    for i, (benchmark, compilers) in enumerate(benchmark_dict.items()):
        x_offset = bar_width / 2 * (1 - n_bars)
        for compiler, (mean, min_val, max_val) in compilers.items():
            mean, min_val, max_val = (
                mean / benchmark_dict[benchmark]["static"][0],
                min_val / benchmark_dict[benchmark]["static"][0],
                max_val / benchmark_dict[benchmark]["static"][0],
            )
            geomeans[compiler] *= mean
            plt.bar(
                i + x_offset,
                mean,
                width=bar_width,
                color=COLOURS[compiler],
                linestyle=LINE_STYLES[compiler],
            )
            plt.vlines(
                i + x_offset,
                min_val,
                max_val,
                color="black",
                linewidth=1,
                linestyle=LINE_STYLES[compiler],
            )
            labels[compiler] = mpatches.Patch(
                color=COLOURS[compiler], linestyle=LINE_STYLES[compiler]
            )
            x_offset += bar_width
    for compiler in geomeans:
        geomeans[compiler] = geomeans[compiler] ** (1 / len(benchmark_dict))
    plt.axvline(len(benchmark_dict) - 0.5, color="gray", linestyle="dotted")
    x_offset = bar_width / 2 * (1 - n_bars)
    for compiler, mean in geomeans.items():
        plt.bar(
            len(benchmark_dict) + x_offset,
            mean,
            width=bar_width,
            color=COLOURS[compiler],
            linestyle=LINE_STYLES[compiler],
        )
        x_offset += bar_width

    plt.xticks(
        range(len(benchmark_dict) + 1),
        list(map(benchmark_short_name, benchmark_dict.keys())) + ["geomean"],
        rotation=45,
    )
    # plt.xlabel('Benchmarks')
    plt.ylabel("Normalized Execution Time")
    plt.xlim(-0.5, len(benchmark_dict) + 0.5)
    # plt.title(f"{compression} - {error_rate} - {code_distance}")
    plt.legend(
        labels.values(),
        list(map(lambda x: COMPILER_MAP[x], labels.keys())),
        loc="upper center",
        ncol=3,
    )
    plt.tight_layout()
    plt.savefig(
        path.join(output_directory, f"{compression}_{error_rate}_{code_distance}.svg")
    )
    plt.close()

    improvement = []
    for benchmark in benchmark_dict:
        improvement.append(
            benchmark_dict[benchmark]["rescq"][0]
            / max(
                benchmark_dict[benchmark]["static"][0],
                benchmark_dict[benchmark]["autobraid"][0],
            )
        )
    geomean = 1
    for imp in improvement:
        geomean *= imp
    geomean = geomean ** (1 / len(improvement))
    with open(speedup_file, "a", encoding="utf-8") as f:
        f.write(f"{compression} {error_rate} {code_distance} {geomean}\n")

    return geomean


def main(input_directory: str, output_directory: str):
    """
    main function of this file
    """
    output_directory = path.join(output_directory, "execution")
    if not path.exists(output_directory):
        makedirs(output_directory)
    speedup_file = path.join(output_directory, "speedup")
    with open(speedup_file, "w", encoding="utf-8") as f:
        f.write("compression error_rate code_distance speedup\n")
    speedup_compression_file = path.join(output_directory, "speedup_compression")
    with open(speedup_compression_file, "w", encoding="utf-8") as f:
        f.write("compression speedup\n")

    for compression in COMPRESSIONS:
        improvement = []
        for error_rate in ERROR_RATES:
            for code_distance in CODE_DISTANCES:
                improvement.append(
                    generate_final_plots(
                        input_directory,
                        output_directory,
                        compression,
                        error_rate,
                        code_distance,
                        speedup_file,
                    )
                )
        geomean = 1
        for imp in improvement:
            geomean *= imp
        geomean = geomean ** (1 / len(improvement))
        with open(speedup_compression_file, "a", encoding="utf-8") as f:
            f.write(f"{compression} {geomean}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("input_directory", type=str)
    args.add_argument("output_directory", type=str)
    args = args.parse_args()
    main(args.input_directory, args.output_directory)
