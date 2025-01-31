import argparse
from collections import defaultdict
from os import makedirs, path
from copy import deepcopy
from multiprocessing import Process
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from cycler import cycler
import pandas as pd
import numpy as np


LINE_STYLES = {"rescq": "-", "static": "--", "autobraid": "-."}
FIXED_COMPRESSIONS = [25 * i for i in range(0, 1)]
COMPRESSIONS = [25 * i for i in range(0, 5)]
COMPILERS = [
    "rescq_25",
    "rescq_50",
    "rescq_100",
    "rescq_200",
    "static",
    "autobraid",
]
COMPILER_MAP = {
    "static": "greedy",
    "autobraid": "AutoBraid",
    "rescq": r"$rescq_{25}$",
}
FIXED_ERROR_RATES = list(range(4, 5))
FIXED_CODE_DISTANCES = [2 * i + 1 for i in range(3, 4)]
ERROR_RATES = list(range(3, 7))
CODE_DISTANCES = [2 * i + 1 for i in range(1, 6)]
MST_FREQUENCY = [25 * (2**i) for i in range(4)]
CYCLER = cycler("color", cm.get_cmap("viridis")([0, 0.4, 0.8])) + cycler(
    "linestyle", LINE_STYLES.values()
)
RESCQ_CYCLER = cycler("color", cm.get_cmap("inferno")([0.1, 0.35, 0.6, 0.85])) + cycler(
    "linestyle", ["-"] * 4
)

BENCHMARK_CLASSES = {
    # "long": {"gates_qft_n320", "gates_vqe_uccsd_n28"},
    "large": {
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
    },
    "medium": {"gates_dnn_n16", "gates_gcm_n13", "gates_qft_n18", "gates_wstate_n27"},
    "supermarq": {
        "gates_HamiltonianSimulation_25",
        "gates_HamiltonianSimulation_50",
        "gates_HamiltonianSimulation_75",
        "gates_QAOAFermionicSwapProxy_15",
        "gates_QAOAVanillaProxy_15",
        "gates_VQEProxy_13",
    },
}


def rescq_sensitivity(
    input_directory: str,
    output_directory: str,
    benchmark_class: str,
    benchmark: str,
    fixed_params: dict,
) -> Dict[int, List[float]]:
    """
    generate the rescq sensitivity plots
    """
    plt.rc("axes", prop_cycle=deepcopy(RESCQ_CYCLER))
    if fixed_params is None:
        fixed_params = {}

    order = ["error_rates", "code_distances"]
    for param in fixed_params.keys():
        order.remove(param)
    error_rates = (
        ERROR_RATES
        if "error_rates" not in fixed_params
        else [fixed_params["error_rates"]]
    )
    code_distances = (
        CODE_DISTANCES
        if "code_distances" not in fixed_params
        else [fixed_params["code_distances"]]
    )

    labels = []
    x_vals = []
    mean_times = defaultdict(list)
    for frequency in MST_FREQUENCY:
        times_of_times = []
        x_vals = []
        for error_rate in error_rates:
            for code_distance in code_distances:
                current_directory = path.join(
                    input_directory,
                    f"{benchmark_class}_rescq_{error_rate}_{code_distance}_{frequency}",
                    benchmark,
                )
                try:
                    f = open(path.join(current_directory, "log"), "r", encoding="utf-8")
                    lines = f.readlines()
                    f.close()
                except FileNotFoundError:
                    print(f"Could not find {path.join(current_directory, 'log')}")
                    return {}
                times = []
                for line in lines:
                    if line.startswith("Done in"):
                        times.append(float(line.split()[2]))
                mean_times[frequency].append(sum(times) / len(times))
                times_of_times.append(times)
                x_vals.append(-error_rate if len(error_rates) > 1 else code_distance)

        plot = plt.violinplot(times_of_times, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle("-")
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (mpatches.Patch(color=plot_colour, linestyle="-"), f"$k = {frequency}$")
        )

    plt.xticks(
        x_vals,
        (
            x_vals
            if len(code_distances) > 1
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel("Code Distance" if len(code_distances) > 1 else "Error Rate")
    plt.ylabel("Num cycles")
    if benchmark == "gates_dnn_n16":
        plt.legend(*zip(*labels), loc="upper center")
    benchmark_name = benchmark[benchmark.find("_") + 1 :]
    plt.title(
        benchmark_name
        + ", "
        + (
            "d = " + str(code_distances[0])
            if len(code_distances) == 1
            else ("p = $10^{" + str(-error_rates[0]) + "}$")
        )
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            benchmark_name
            + "_mst"
            + (
                "_d" + str(code_distances[0])
                if len(code_distances) == 1
                else "_p" + str(error_rates[0])
            )
            + ".svg",
        )
    )
    plt.close()
    plt.clf()

    return mean_times


def sensitivity(
    input_directory: str,
    output_directory: str,
    benchmark_class: str,
    benchmark: str,
    fixed_params: dict,
) -> Dict[str, List[float]]:
    """
    generate the sensitivity plots
    """
    plt.rc("axes", prop_cycle=deepcopy(CYCLER))
    if fixed_params is None:
        fixed_params = {}

    order = ["error_rates", "code_distances"]
    for param in fixed_params.keys():
        order.remove(param)
    error_rates = (
        ERROR_RATES
        if "error_rates" not in fixed_params
        else [fixed_params["error_rates"]]
    )
    code_distances = (
        CODE_DISTANCES
        if "code_distances" not in fixed_params
        else [fixed_params["code_distances"]]
    )

    labels = []
    x_vals = []
    mean_times = defaultdict(list)
    for compilerstr in COMPILERS:
        compiler, frequency = "", ""
        if compilerstr.find("_") != -1:
            compiler, frequency = compilerstr.split("_")
        else:
            compiler = compilerstr
        if frequency not in ("", "25"):
            continue
        times_of_times = []
        x_vals = []
        for error_rate in error_rates:
            for code_distance in code_distances:
                if frequency != "":
                    current_directory = path.join(
                        input_directory,
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}_{frequency}",
                        benchmark,
                    )
                else:
                    current_directory = path.join(
                        input_directory,
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}",
                        benchmark,
                    )
                try:
                    f = open(path.join(current_directory, "log"), "r", encoding="utf-8")
                    lines = f.readlines()
                    f.close()
                except FileNotFoundError:
                    print(f"Could not find {path.join(current_directory, 'log')}")
                    return {}
                times = []
                for line in lines:
                    if line.startswith("Done in"):
                        times.append(float(line.split()[2]))
                mean_times[compiler].append(sum(times) / len(times))
                times_of_times.append(times)
                x_vals.append(-error_rate if len(error_rates) > 1 else code_distance)

        plot = plt.violinplot(times_of_times, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(
        x_vals,
        (
            x_vals
            if len(code_distances) > 1
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel("Code Distance" if len(code_distances) > 1 else "Error Rate")
    plt.ylabel("Num cycles")
    if benchmark == "gates_dnn_n16":
        if len(code_distances) > 1:
            plt.legend(*zip(*labels), loc="upper right")
        else:
            plt.legend(*zip(*labels), loc="center left")
    benchmark_name = benchmark[benchmark.find("_") + 1 :]
    plt.title(
        benchmark_name
        + ", "
        + (
            "d = " + str(code_distances[0])
            if len(code_distances) == 1
            else ("p = $10^{" + str(-error_rates[0]) + "}$")
        )
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            benchmark_name
            + (
                "_d" + str(code_distances[0])
                if len(code_distances) == 1
                else "_p" + str(error_rates[0])
            )
            + ".svg",
        )
    )
    plt.close()
    plt.clf()

    return mean_times


def idling(
    input_directory: str,
    output_directory: str,
    benchmark_class: str,
    benchmark: str,
    fixed_params: dict,
) -> Dict[str, List[float]]:
    """
    generate the idling plots
    """
    plt.rc("axes", prop_cycle=deepcopy(CYCLER))
    if fixed_params is None:
        fixed_params = {}

    order = ["error_rates", "code_distances"]
    for param in fixed_params.keys():
        order.remove(param)
    error_rates = (
        ERROR_RATES
        if "error_rates" not in fixed_params
        else [fixed_params["error_rates"]]
    )
    code_distances = (
        CODE_DISTANCES
        if "code_distances" not in fixed_params
        else [fixed_params["code_distances"]]
    )

    labels = []
    x_vals = []
    mean_times = defaultdict(list)
    for compilerstr in COMPILERS:
        compiler, frequency = "", ""
        if compilerstr.find("_") != -1:
            compiler, frequency = compilerstr.split("_")
        else:
            compiler = compilerstr
        if frequency not in ("", "25"):
            continue
        times_of_times = []
        x_vals = []
        for error_rate in error_rates:
            for code_distance in code_distances:
                if frequency != "":
                    current_directory = path.join(
                        input_directory,
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}_{frequency}",
                        benchmark,
                    )
                else:
                    current_directory = path.join(
                        input_directory,
                        f"{benchmark_class}_{compiler}_{error_rate}_{code_distance}",
                        benchmark,
                    )
                try:
                    f = open(path.join(current_directory, "log"), "r", encoding="utf-8")
                    lines = f.readlines()
                    f.close()
                except FileNotFoundError:
                    print(f"Could not find {path.join(current_directory, 'log')}")
                    return {}
                total_time = 0
                num_runs = 0
                for line in lines:
                    if line.startswith("Done in"):
                        total_time += float(line.split()[2])
                        num_runs += 1
                df = pd.read_csv(
                    path.join(current_directory, "dataq_heatmap.csv"),
                    header=None,
                    dtype=np.float64,
                )
                df = df.loc[:, (df > 0).any()]
                times = 1 - num_runs * df.sum(axis=0) / total_time
                mean_times[compiler].append(times.mean())
                times_of_times.append(times.tolist())
                x_vals.append(-error_rate if len(error_rates) > 1 else code_distance)

        plot = plt.violinplot(times_of_times, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(
        x_vals,
        (
            x_vals
            if len(code_distances) > 1
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel("Code Distance" if len(code_distances) > 1 else "Error Rate")
    plt.ylabel("Fraction of qubit idle time")
    if benchmark == "gates_dnn_n16":
        plt.legend(*zip(*labels), loc="center left")
    benchmark_name = benchmark[benchmark.find("_") + 1 :]
    plt.title(
        benchmark_name
        + ", "
        + (
            "d = " + str(code_distances[0])
            if len(code_distances) == 1
            else ("p = $10^{" + str(-error_rates[0]) + "}$")
        )
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            benchmark_name
            + "_idling"
            + (
                "_d" + str(code_distances[0])
                if len(code_distances) == 1
                else "_p" + str(error_rates[0])
            )
            + ".svg",
        )
    )
    plt.close()
    plt.clf()

    return mean_times


def codesign_sensitivity(
    input_directory: str,
    output_directory: str,
    benchmark_class: str,
    benchmark: str,
    error_rate: int,
    code_distance: int,
) -> Dict[str, List[float]]:
    """
    generate the codesign sensitivity plots
    """
    plt.rc("axes", prop_cycle=deepcopy(CYCLER))
    labels = []
    x_vals = []
    mean_times = defaultdict(list)
    for compilerstr in COMPILERS:
        compiler, frequency = "", ""
        if compilerstr.find("_") != -1:
            compiler, frequency = compilerstr.split("_")
        else:
            compiler = compilerstr
        if frequency not in ("", "25"):
            continue
        times_of_times = []
        x_vals = []
        for compression in COMPRESSIONS:
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
            try:
                f = open(path.join(current_directory, "log"), "r", encoding="utf-8")
                lines = f.readlines()
                f.close()
            except FileNotFoundError:
                print(f"Could not find {path.join(current_directory, 'log')}")
                return {}
            times = []
            for line in lines:
                if line.startswith("Done in"):
                    times.append(float(line.split()[2]))
            mean_times[compiler].append(sum(times) / len(times))
            times_of_times.append(times)
            x_vals.append(compression / 25)

        plot = plt.violinplot(times_of_times, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(x_vals, list(map(str, COMPRESSIONS)))
    plt.xlabel("Grid Compression Percentage")
    plt.ylabel("Num cycles")
    if benchmark == "gates_dnn_n16":
        plt.legend(*zip(*labels), loc="upper left")
    benchmark_name = benchmark[benchmark.find("_") + 1 :]
    plt.title(
        benchmark_name
        + ", "
        + ("d = " + str(code_distance) + ", p = $10^{" + str(-error_rate) + "}$")
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            benchmark_name
            + "_d"
            + str(code_distance)
            + "_p"
            + str(error_rate)
            + ".svg",
        )
    )
    plt.close()

    return mean_times


def all_benchmarks(input_directory: str, output_directory: str, fixed_params: dict):
    """
    generate all benchmarks
    """
    plt.box(False)
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (9, 6)
    sensitivity_values: Dict[str, List[List[float]]] = defaultdict(
        lambda: [
            []
            for _ in range(
                len(CODE_DISTANCES)
                if "code_distances" not in fixed_params
                else len(ERROR_RATES)
            )
        ]
    )
    idling_values: Dict[str, List[List[float]]] = defaultdict(
        lambda: [
            []
            for _ in range(
                len(CODE_DISTANCES)
                if "code_distances" not in fixed_params
                else len(ERROR_RATES)
            )
        ]
    )
    rescq_values: Dict[int, List[List[float]]] = defaultdict(
        lambda: [
            []
            for _ in range(
                len(CODE_DISTANCES)
                if "code_distances" not in fixed_params
                else len(ERROR_RATES)
            )
        ]
    )
    for benchmark_class, benchmarks in BENCHMARK_CLASSES.items():
        for benchmark in benchmarks:
            sensitivity_value = sensitivity(
                input_directory,
                output_directory,
                benchmark_class,
                benchmark,
                fixed_params,
            )
            idling_value = idling(
                input_directory,
                output_directory,
                benchmark_class,
                benchmark,
                fixed_params,
            )
            rescq_value = rescq_sensitivity(
                input_directory,
                output_directory,
                benchmark_class,
                benchmark,
                fixed_params,
            )
            for compiler, values in sensitivity_value.items():
                for i, value in enumerate(values):
                    sensitivity_values[compiler][i].append(value)
            for compiler, values in idling_value.items():
                for i, value in enumerate(values):
                    idling_values[compiler][i].append(value)
            for frequency, values in rescq_value.items():
                for i, value in enumerate(values):
                    rescq_values[frequency][i].append(value)

    normalization = sensitivity_values["static"][0].copy()
    for compiler, values in sensitivity_values.items():
        for i, benchmark_vals in enumerate(values):
            for j, val in enumerate(benchmark_vals):
                benchmark_vals[j] = val / normalization[j]

    normalization = rescq_values[MST_FREQUENCY[-1]][0].copy()
    for frequency, values in rescq_values.items():
        for i, benchmark_vals in enumerate(values):
            for j, val in enumerate(benchmark_vals):
                benchmark_vals[j] = val / normalization[j]

    plt.rc("axes", prop_cycle=deepcopy(CYCLER))
    labels = []
    x_vals = []
    for compiler, values in sensitivity_values.items():
        x_vals = (
            CODE_DISTANCES
            if "code_distances" not in fixed_params
            else list(map(lambda x: -x, ERROR_RATES))
        )
        plot = plt.violinplot(values, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(
        x_vals,
        (
            list(map(str, x_vals))
            if "code_distances" not in fixed_params
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel(
        "Code Distance" if "code_distances" not in fixed_params else "Error Rate"
    )
    plt.ylabel("Relative time")
    # plt.legend(*zip(*labels), loc="upper right")
    plt.title(
        "All benchmarks (relative to greedy for "
        + (
            f"p = $10^{{{-ERROR_RATES[0]}}}$"
            if "code_distances" in fixed_params
            else f"d = {CODE_DISTANCES[0]}"
        )
        + ")"
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            "all"
            + (
                "_d" + str(fixed_params["code_distances"])
                if "code_distances" in fixed_params
                else "_p" + str(fixed_params["error_rates"])
            )
            + ".svg",
        )
    )
    plt.close()

    idling_cycler = cycler("color", cm.get_cmap("viridis")([0, 0.4, 0.8])) + cycler(
        "linestyle", LINE_STYLES.values()
    )
    plt.rc("axes", prop_cycle=deepcopy(idling_cycler))
    labels = []
    x_vals = []
    for compiler, values in idling_values.items():
        x_vals = (
            CODE_DISTANCES
            if "code_distances" not in fixed_params
            else list(map(lambda x: -x, ERROR_RATES))
        )
        plot = plt.violinplot(values, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(
        x_vals,
        (
            list(map(str, x_vals))
            if "code_distances" not in fixed_params
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel(
        "Code Distance" if "code_distances" not in fixed_params else "Error Rate"
    )
    plt.ylabel("Fraction of qubit idle time")
    # plt.legend(*zip(*labels), loc="upper right")
    plt.title("All benchmarks")
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            "all_idling"
            + (
                "_d" + str(fixed_params["code_distances"])
                if "code_distances" in fixed_params
                else "_p" + str(fixed_params["error_rates"])
            )
            + ".svg",
        )
    )
    plt.close()

    plt.rc("axes", prop_cycle=deepcopy(RESCQ_CYCLER))
    labels = []
    x_vals = []
    for frequency, values in rescq_values.items():
        x_vals = (
            CODE_DISTANCES
            if "code_distances" not in fixed_params
            else list(map(lambda x: -x, ERROR_RATES))
        )
        plot = plt.violinplot(values, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle("-")
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (mpatches.Patch(color=plot_colour, linestyle="-"), f"$k = {frequency}$")
        )

    plt.xticks(
        x_vals,
        (
            list(map(str, x_vals))
            if "code_distances" not in fixed_params
            else list(map(lambda x: "$10^{" + str(x) + "}$", x_vals))
        ),
    )
    plt.xlabel(
        "Code Distance" if "code_distances" not in fixed_params else "Error Rate"
    )
    plt.ylabel("Relative time")
    # plt.legend(*zip(*labels), loc="upper right")
    plt.title(
        f"All benchmarks (relative to $k = {MST_FREQUENCY[-1]}$ for "
        + (
            f"p = $10^{{{-ERROR_RATES[0]}}}$"
            if "code_distances" in fixed_params
            else f"d = {CODE_DISTANCES[0]}"
        )
        + ")"
    )
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            "all_mst"
            + (
                "_d" + str(fixed_params["code_distances"])
                if "code_distances" in fixed_params
                else "_p" + str(fixed_params["error_rates"])
            )
            + ".svg",
        )
    )


def all_codesign(
    input_directory: str, output_directory: str, error_rate: int, code_distance: int
):
    """
    generate all codesign plots
    """
    plt.box(False)
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (9, 6)
    sensitivity_values: Dict[str, List[List[float]]] = defaultdict(
        lambda: [[] for _ in range(len(COMPRESSIONS))]
    )
    for benchmark_class, benchmarks in BENCHMARK_CLASSES.items():
        for benchmark in benchmarks:
            sensitivity_value = codesign_sensitivity(
                input_directory,
                output_directory,
                benchmark_class,
                benchmark,
                error_rate,
                code_distance,
            )
            for compiler, values in sensitivity_value.items():
                for i, value in enumerate(values):
                    sensitivity_values[compiler][i].append(value)

    normalization = sensitivity_values["static"][-1].copy()
    for compiler, values in sensitivity_values.items():
        for i, benchmark_vals in enumerate(values):
            for j, val in enumerate(benchmark_vals):
                benchmark_vals[j] = val / normalization[j]

    plt.rc("axes", prop_cycle=deepcopy(CYCLER))
    labels = []
    x_vals = []
    for compiler, values in sensitivity_values.items():
        x_vals = list(map(lambda x: x / 25, COMPRESSIONS))
        plot = plt.violinplot(values, positions=x_vals, showmeans=True)
        plot_colour = (
            plot["bodies"][0]
            .get_facecolor(  # type: ignore
            )
            .flatten()
        )
        for body in plot["bodies"]:  # type: ignore
            body.set_linestyle(LINE_STYLES[compiler])
            body.set_linewidth(1)
            body.set_edgecolor("black")
        labels.append(
            (
                mpatches.Patch(color=plot_colour, linestyle=LINE_STYLES[compiler]),
                COMPILER_MAP[compiler],
            )
        )

    plt.xticks(x_vals, list(map(str, COMPRESSIONS)))
    plt.xlabel("Grid Compression Percentage")
    plt.ylabel("Relative time")
    # plt.legend(*zip(*labels), loc="upper left")
    plt.title(f"All benchmarks (relative to greedy at ${COMPRESSIONS[-1]}\\%$)")
    plt.tight_layout()
    plt.savefig(
        path.join(
            output_directory,
            "all" + "_d" + str(code_distance) + "_p" + str(error_rate) + ".svg",
        )
    )
    plt.close()


def main(input_directory: str, output_directory: str, compression: int):
    """
    main function for this file
    """
    input_directory = path.join(input_directory, str(compression))
    output_directory = path.join(output_directory, "sensitivity_" + str(compression))
    if not path.exists(output_directory):
        makedirs(output_directory)

    procs = []
    for error_rate in FIXED_ERROR_RATES:
        proc = Process(
            target=all_benchmarks,
            args=(input_directory, output_directory, {"error_rates": error_rate}),
        )
        proc.start()
        procs.append(proc)
    for code_distance in FIXED_CODE_DISTANCES:
        proc = Process(
            target=all_benchmarks,
            args=(input_directory, output_directory, {"code_distances": code_distance}),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def do_all(input_directory: str, output_directory: str):
    """
    do all the compressions
    """
    for compression in FIXED_COMPRESSIONS:
        main(input_directory, output_directory, compression)

    output_directory = path.join(output_directory, "sensitivity_codesign")
    if not path.exists(output_directory):
        makedirs(output_directory)
    procs = []
    for error_rate in FIXED_ERROR_RATES:
        for code_distance in FIXED_CODE_DISTANCES:
            proc = Process(
                target=all_codesign,
                args=(input_directory, output_directory, error_rate, code_distance),
            )
            proc.start()
            procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("input_directory", type=str)
    args.add_argument("output_directory", type=str)
    args.add_argument("compression", type=int, default=None, nargs="?")
    args = args.parse_args()
    if args.compression is None:
        do_all(args.input_directory, args.output_directory)
    else:
        main(args.input_directory, args.output_directory, args.compression)
