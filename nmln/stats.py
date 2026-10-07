import math
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcNumRings


def numBonds(MOLS):
    num_bonds = []
    for mol in MOLS:
        num_bonds.append(mol.GetNumBonds())
    return num_bonds


def maxNodeDegree(MOLS):
    num_bonds = []
    for mol in MOLS:
        avg = []
        for a in mol.GetAtoms():
            avg.append(a.GetDegree())
        num_bonds.append(float(np.max(avg)) + 1e-12)
    return num_bonds


def minNodeDegree(MOLS):
    num_bonds = []
    for mol in MOLS:
        avg = []
        for a in mol.GetAtoms():
            avg.append(a.GetDegree())
        num_bonds.append(float(np.min(avg)) + 1e-12)
    return num_bonds


def avgNodeDegree(MOLS):
    num_bonds = []
    for mol in MOLS:
        avg = []
        for a in mol.GetAtoms():
            avg.append(a.GetDegree())
        num_bonds.append(float(sum(avg)) / (float(len(avg)) + 1e-12))
    return num_bonds


def calcHistDiff(f, generated_molecules, true_molecules, bins, bw, ax=None, title=None):
    g = f(generated_molecules)
    t = f(true_molecules)

    gh = np.histogram(g, bins, density=True)[0]
    th = np.histogram(t, bins, density=True)[0]
    diff = abs(gh - th).sum()
    return diff


def plotHistComparison(
    f, generated_molecules, true_molecules, bins, bw, ax=None, title=None
):
    g = f(generated_molecules)
    t = f(true_molecules)

    if not g:
        diff = float("inf")

    elif not t:
        diff = float("inf")

    else:
        diff = calcHistDiff(f, generated_molecules, true_molecules, bins, bw)

    sns.histplot(
        g,
        element="step",
        stat="density",
        linewidth=2,
        alpha=0.1,
        color="blue",
        bins=bins,
        label="generated",
        ax=ax,
    )

    sns.histplot(
        t,
        element="step",
        stat="density",
        linewidth=2,
        linestyle="--",
        alpha=0.1,
        color="red",
        bins=bins,
        label="training",
        ax=ax,
    )

    title = (f.__name__ if title is None else title) + f" (diff {diff:.4f})"

    if ax is None:
        plt.title(title)
        plt.show()
    else:
        ax.set_title(title)

    return diff


def calcNumHBD(x):
    return [CalcNumHBD(a) for a in x]


def calcNumRings(x):
    return [CalcNumRings(a) for a in x]


def calcNumHBA(x):
    return [CalcNumHBA(a) for a in x]


def calcTPSA(x):
    return [CalcTPSA(a) for a in x]


FS = [  # (numAtoms,np.arange(0,12,1),0.5),
    (numBonds, np.arange(0, 20, 1), 0.5, "Number of Bonds"),
    (avgNodeDegree, np.arange(0, 5, 0.5), 0.5, "Average Node Degree"),
    (calcNumRings, np.arange(0, 4, 1), 0.5, "Number of Rings"),
    (calcNumHBD, np.arange(0, 7, 1), 0.5, "HBD"),
    (calcNumHBA, np.arange(0, 10, 1), 0.5, "HBA"),
    (calcTPSA, np.arange(0, 100, 10), 5.0, "TPSA"),
    (maxNodeDegree, np.arange(0, 5, 0.5), 0.5, "Max Node Degree"),
    (minNodeDegree, np.arange(0, 5, 0.5), 0.5, "Min Node Degree"),
]


def diffs_histograms_image(generated_molecules, true_molecules):
    output = {}
    for i, f in enumerate(FS):
        diff = calcHistDiff(
            f[0],
            generated_molecules=generated_molecules,
            true_molecules=true_molecules,
            bins=f[1],
            bw=f[2],
        )
        output[f[3]] = diff
    return output


def stats_histograms_image(filename, generated_molecules, true_molecules):
    summary = {}
    plt.clf()
    fig = plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
    axs = []
    for i, f in enumerate(tqdm(FS, desc="FS")):
        ax = plt.subplot(int(math.ceil(len(FS) / 3)), 3, i + 1)
        diff = plotHistComparison(
            f[0],
            generated_molecules=generated_molecules,
            true_molecules=true_molecules,
            bins=f[1],
            bw=f[2],
            ax=ax,
            title=f[3],
        )
        summary[f[3]] = diff
        axs.append(ax)

    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="lower center", ncol=5, labelspacing=0.0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    fig.canvas.draw()
    plt.savefig(filename)
    plt.close(fig)

    return summary
