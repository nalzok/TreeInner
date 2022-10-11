from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import click


def visualize(results: Path, data_root: Path, agg_by: str):
    sns.set_theme(style="whitegrid", rc={"savefig.dpi": 300})
    log_xscale = {"eta", "reg_lambda"}

    oracle_auc = pd.read_csv(results / "csv" / f"auc-by-{agg_by}-{data_root}.csv")
    oracle_auc["style"] = oracle_auc["ifa"] + "." + oracle_auc["domain"]

    # by = ["distribution", "subproblem", agg_by, "gfa", "ifa", "domain"]
    # auc_mean_std = oracle_auc.groupby(by).agg(
    #     auc_mean=("auc", "mean"),
    #     auc_std=("auc", "std"),
    #     risk_valid_mean=("risk_valid", "mean"),
    #     risk_valid_std=("risk_valid", "std"),
    # )
    # print(auc_mean_std.to_latex())

    for metric in ("auc", "score_noisy", "score_signal"):
        fig_name = results / "plots" / f"{metric}-by-{agg_by}-{data_root}.png"
        print("Creating", fig_name)
        facet_kws = {}
        if metric == "auc":
            facet_kws["ylim"] = (0, 1)
        else:
            facet_kws["sharey"] = False

        g = sns.relplot(
            data=oracle_auc,
            x=agg_by,
            y=metric,
            row="distribution",
            col="subproblem",
            kind="line",
            style="style",
            hue="gfa",
            markers={
                "PreDecomp.in": "o",
                "PreDecomp.out": "o",
                "SHAP.in": "X",
                "SHAP.out": "X",
                "Permutation.in": "*",
                "Permutation.out": "*",
            },
            dashes={
                "PreDecomp.in": (2, 1),
                "SHAP.in": (2, 1),
                "Permutation.in": (2, 1),
                "PreDecomp.out": "",
                "SHAP.out": "",
                "Permutation.out": "",
            },
            errorbar=None,
            height=8,
            aspect=2,
            facet_kws=facet_kws,
        )
        if agg_by in log_xscale:
            g.set(xscale="log")
        g.fig.suptitle(f"{metric} by {agg_by} on {data_root}")
        g.savefig(fig_name)

    fig_name = results / "plots" / f"risk-by-{agg_by}-{data_root}.png"
    print("Creating", fig_name)
    facet_kws = {}
    if metric == "auc":
        facet_kws["ylim"] = (0, 1)
    else:
        facet_kws["sharey"] = False

    g = sns.relplot(
        data=oracle_auc,
        x=agg_by,
        y="risk_valid",
        row="distribution",
        col="subproblem",
        kind="line",
        height=8,
        aspect=2,
        facet_kws=facet_kws,
    )
    if agg_by in log_xscale:
        g.set(xscale="log")
    g.fig.suptitle(f"risk by {agg_by} on {data_root}")
    g.savefig(fig_name)

    mdi_error = pd.read_csv(results / "csv" / f"error-by-{agg_by}-{data_root}.csv")
    mdi_error["log10(rel_error)"] = np.log10(mdi_error["rel_error"])

    fig_name = results / "plots" / f"error-by-{agg_by}-{data_root}.png"
    print("Creating", fig_name)
    g = sns.catplot(
        data=mdi_error,
        x=agg_by,
        y="log10(rel_error)",
        row="distribution",
        col="subproblem",
        kind="box",
    )
    g.savefig(fig_name)


@click.command()
@click.option("--results", type=click.Path(path_type=Path), default="results")
@click.option(
    "--data_root", type=click.Path(path_type=Path), default="04_aggregate_2019"
)
@click.option("--agg_by", type=str, required=True)
def cli(results: Path, data_root: Path, agg_by: str):
    visualize(results, data_root, agg_by)


if __name__ == "__main__":
    cli()
