from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import click


def visualize(results: Path, agg_by: str):
    sns.set_theme(style="whitegrid", rc={"savefig.dpi": 300})

    oracle_auc = pd.read_csv(results / "csv" / f"auc-by-{agg_by}.csv")

    by = ["subproblem", "subproblem_id", agg_by, "gfa", "ifa", "domain"]
    auc_mean_std = oracle_auc.groupby(by).agg(
        auc_noisy_mean=("auc_noisy", "mean"),
        auc_noisy_std=("auc_noisy", "std"),
        risk_valid_mean=("risk_valid", "mean"),
        risk_valid_std=("risk_valid", "std"),
    )
    # print(auc_mean_std.to_latex())

    fig_name = results / "plots" / f"auc-by-{agg_by}.png"
    print("Creating", fig_name)
    g = sns.relplot(
        data=oracle_auc,
        x=agg_by,
        y="auc_noisy",
        row="subproblem",
        col="subproblem_id",
        kind="line",
        style="domain",
        hue="ifa",
        size="gfa",
        sizes={"Abs": 1, "Inner": 4},
        errorbar=None,
        height=16,
        aspect=2,
        facet_kws=dict(ylim=(0, 1))
    )
    # g.map(sns.lineplot, agg_by, "risk_valid", color="pink", errorbar=None, lw=3)
    g.savefig(fig_name)

    fig_name = results / "plots" / f"auc-risk-by-{agg_by}.png"
    print("Creating", fig_name)
    g = sns.relplot(
        data=auc_mean_std,
        x="risk_valid_mean",
        y="auc_noisy_mean",
        row="subproblem",
        col="subproblem_id",
        kind="scatter",
        style="domain",
        hue="ifa",
        size="gfa",
        sizes={"Abs": 100, "Inner": 200},
        height=16,
        aspect=2,
        facet_kws=dict(sharex=False)
    )
    g.savefig(fig_name)

    mdi_error = pd.read_csv(results / "csv" / f"error-by-{agg_by}.csv")
    mdi_error["log10(error)"] = np.log10(mdi_error["error"])

    fig_name = results / "plots" / f"error-by-{agg_by}.png"
    print("Creating", fig_name)
    g = sns.catplot(
        data=mdi_error,
        x=agg_by,
        y="log10(error)",
        row="subproblem",
        col="subproblem_id",
        kind="box",
    )
    g.savefig(fig_name)


@click.command()
@click.option("--results", type=click.Path(path_type=Path), default="results")
@click.option("--agg_by", type=str, required=True)
def cli(results, agg_by):
    visualize(results, agg_by)


if __name__ == "__main__":
    cli()
