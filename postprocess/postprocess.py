import os
from shutil import rmtree
from sys import argv
from sys import exit as sys_exit
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
import numpy as np


def get_dims(dirpath: str) -> tuple[int, int, int]:
    with open(os.path.join(dirpath, "log"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    numqubits, numrows, numcols = -1, -1, -1
    for line in lines:
        if line.startswith("numQubits"):
            numqubits = int(line.split()[1])
        elif line.startswith("numRows"):
            numrows = int(line.split()[1])
        elif line.startswith("numColumns"):
            numcols = int(line.split()[1])
            assert numqubits != -1 and numrows != -1 and numcols != -1
            break
    return numqubits, numrows, numcols


def generate_heatmap(
    dirpath: str, filename: str, grid_rows: int, grid_cols: int
) -> None:
    df = read_csv(os.path.join(dirpath, filename), header=None, dtype=np.float64)

    dirpath = os.path.join(dirpath, filename.split("_")[0])
    if os.path.exists(dirpath):
        rmtree(dirpath)
    os.mkdir(dirpath)

    for i, row in df.iterrows():
        row = np.array(row).reshape(grid_rows, grid_cols)
        row[row < 0] = np.nan
        ax = plt.axes()
        ax.set_facecolor("#d9d9d9")
        sns.heatmap(row, cmap="hot_r", vmax=500)
        plt.title(f"Epoch {i}")
        plt.savefig(os.path.join(dirpath, f"{i}.png"))
        plt.clf()


def plot_histogram(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    filename: str,
    mean: float | None,
    tail: float | None,
) -> None:
    plt.stairs(hist, bin_edges, fill=True)
    if mean is not None:
        plt.axvline(mean, color="brown", linestyle="dashed", linewidth=1)
    if tail is not None:
        plt.axvline(tail, color="red", linestyle="dashed", linewidth=1)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


def plot_histogram_and_cumulative(dirpath: str, filename: str) -> None:
    df = read_csv(
        os.path.join(dirpath, "logs", filename), header=None, dtype=np.float64
    )
    df = df.sort_values(df.columns[0])
    num_entries = int(df.shape[0] * 0.95)
    tail_avg = df.iloc[num_entries:][0].mean()
    df = df.iloc[:num_entries]

    hist, bin_edges = np.histogram(df, bins="auto")
    plot_histogram(
        hist,
        bin_edges,
        filename.split(".")[0],
        os.path.join(dirpath, "histograms", filename.split("/")[-1] + ".png"),
        df.iloc[:, 0].mean(),
        tail_avg,
    )

    cumsum = np.cumsum(hist)
    plot_histogram(
        cumsum,
        bin_edges,
        filename.split(".")[0],
        os.path.join(dirpath, "cumulative", filename.split("/")[-1] + ".png"),
        None,
        None,
    )


def plot_execution_times(dirpath: str) -> None:
    with open(os.path.join(dirpath, "log"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    times = []
    for line in lines:
        if line.startswith("Done in"):
            times.append(float(line.split()[2]))

    hist, bin_edges = np.histogram(times, bins="auto")
    plot_histogram(
        hist,
        bin_edges,
        "Execution Times",
        os.path.join(dirpath, "histograms", "execution_times.png"),
        float(np.mean(times)),
        None,
    )


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python postprocess.py <output_dir>")
        sys_exit(1)
    OUTPUT_DIR = argv[1]
    print(f"Output directory: {OUTPUT_DIR}")
    if not os.path.isdir(OUTPUT_DIR):
        print("Error: output directory does not exist")
        sys_exit(1)

    for dir_path in os.listdir(OUTPUT_DIR):
        if not os.path.isdir(os.path.join(OUTPUT_DIR, dir_path)):
            continue
        print(f"Processing {dir_path}")
        dir_path = os.path.join(OUTPUT_DIR, dir_path)
        num_qubits, num_rows, num_cols = get_dims(dir_path)
        generate_heatmap(dir_path, "dataq_heatmap.csv", num_rows, num_cols)
        generate_heatmap(dir_path, "ancillaq_heatmap.csv", num_rows, num_cols)

        if os.path.exists(os.path.join(dir_path, "histograms")):
            rmtree(os.path.join(dir_path, "histograms"))
        os.mkdir(os.path.join(dir_path, "histograms"))
        if os.path.exists(os.path.join(dir_path, "cumulative")):
            rmtree(os.path.join(dir_path, "cumulative"))
        os.mkdir(os.path.join(dir_path, "cumulative"))
        plot_execution_times(dir_path)
        for file in os.listdir(os.path.join(dir_path, "logs")):
            plot_histogram_and_cumulative(dir_path, file)
