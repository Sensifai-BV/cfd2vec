"""Pilot figures: training curves and a pedestrian-height field comparison on one test case."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curves(runs: dict, out: str, prior_baseline: dict, style=None):
    """runs: label -> log.jsonl path. Two panels: loss vs step; held-out fully masked U / Cp error vs step."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    colors = {"masked field modelling": "#1f5fa8", "supervised regression (control)": "#9a6b3d"}
    for lab, path in runs.items():
        tr_by, ev_by = {}, {}
        for L in open(path):
            d = json.loads(L)
            (ev_by if "zero_shot" in d else tr_by)[d["step"]] = d     # a resumed run re-logs steps: keep the last record
        tr, ev = [tr_by[k] for k in sorted(tr_by)], [ev_by[k] for k in sorted(ev_by)]
        c = colors.get(lab, "k")
        axs[0].plot([d["step"] for d in tr], [d["loss"] for d in tr], color=c, lw=0.8, alpha=0.8, label=lab)
        axs[0].plot([d["step"] for d in ev], [d["val_recon_loss"] for d in ev], color=c, lw=1.4, ls="--", marker="o", ms=3)
        for ax, key in ((axs[1], "rel_l2_U"), (axs[2], "rel_l2_Cp")):
            ax.plot([d["step"] for d in ev], [d["zero_shot"][key] for d in ev], color=c, marker="o", ms=3, lw=1.4)
            ax.plot([d["step"] for d in ev], [d["with_prior"][key] for d in ev], color=c, marker="s", ms=3, lw=1.0, ls=":")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax, key, name in ((axs[1], "rel_l2_U", "U"), (axs[2], "rel_l2_Cp", "Cp")):
        ax.axhline(prior_baseline[key], color="#777777", lw=0.8, ls="-.")
        ax.text(0.03, prior_baseline[key] * 0.985, "coarse prior alone", transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=6, color="#555555")
        ax.set_xlabel("step"); ax.set_ylabel(f"held-out relative L2 of {name}")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v * 10, 6) in (2, 3, 4, 6) else ""))
    axs[0].set_xlabel("step"); axs[0].set_ylabel("masked-field loss (standardised MSE)")
    axs[0].set_title("Training (solid), held-out (dashed)", loc="left", fontsize=7)
    axs[1].set_title("Velocity", loc="left", fontsize=7)
    axs[2].set_title("Pressure coefficient", loc="left", fontsize=7)
    axs[0].legend(frameon=False, fontsize=6, loc="upper right")
    h = [plt.Line2D([], [], color="#444444", marker="o", ms=3, lw=1.4, label="geometry only (fully masked)"),
         plt.Line2D([], [], color="#444444", marker="s", ms=3, lw=1.0, ls=":", label="with coarse-twin prior")]
    axs[1].legend(handles=h, frameon=False, fontsize=6, loc="upper right")
    fig.savefig(out, dpi=300)
    return fig


def pedestrian_slice(case, preds: dict, out: str, z_target=0.15):
    """|U| / U_ref on the cell layer closest to z_target * H: truth, predictions, and prediction - truth."""
    z = case.points[:, 2]
    layer = np.unique(np.round(z, 5)); zl = layer[np.argmin(np.abs(layer - z_target))]
    m = np.abs(z - zl) < 1e-4
    x, y = case.points[m, 0], case.points[m, 1]
    sp = lambda f: np.linalg.norm(f[m, 0:3], axis=1)  # noqa: E731
    truth = sp(case.fields)
    n = len(preds)
    fig, axs = plt.subplots(2, n + 1, figsize=(7.2, 2.35), constrained_layout=True, sharex=True, sharey=True)
    vmax = np.percentile(truth, 99.5)
    kw = dict(s=1.2, marker="s", lw=0, rasterized=True)
    axs[0, 0].scatter(x, y, c=truth, cmap="viridis", vmin=0, vmax=vmax, **kw); axs[0, 0].set_title("OpenFOAM (truth)", loc="left", fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].text(0.5, 0.5, f"cell layer z = {zl:.2f} H", ha="center", va="center", fontsize=7, transform=axs[1, 0].transAxes)
    for i, (lab, f) in enumerate(preds.items(), start=1):
        sc = axs[0, i].scatter(x, y, c=sp(f), cmap="viridis", vmin=0, vmax=vmax, **kw)
        err = sp(f) - truth
        ec = axs[1, i].scatter(x, y, c=err, cmap="RdBu_r", vmin=-0.3, vmax=0.3, **kw)
        rel = np.linalg.norm(f[m, 0:3] - case.fields[m, 0:3]) / np.linalg.norm(case.fields[m, 0:3])
        axs[0, i].set_title(lab, loc="left", fontsize=7)
        axs[1, i].set_title(f"error, layer rel. L2 = {rel:.2f}", loc="left", fontsize=7)
    for ax in axs.ravel():
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axs[0, :], shrink=0.8, label="|U| / U_ref")
    fig.colorbar(ec, ax=axs[1, 1:], shrink=0.8, label="prediction - truth")
    fig.supxlabel("streamwise x / H", fontsize=7); fig.supylabel("spanwise y / H", fontsize=7)
    fig.savefig(out, dpi=300)
    return fig, float(zl)
