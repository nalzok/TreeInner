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
        auc_mean=("auc_noisy", "mean"), auc_std=("auc_noisy", "std")
    )
    print(auc_mean_std.to_latex())

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
        errorbar=None,
        height=16,
        aspect=2,
    )
    # g.map(sns.lineplot, agg_by, "risk_valid", color="pink", errorbar=None, lw=3)
    g.savefig(results / "plots" / f"auc-by-{agg_by}.png")


    mdi_error = pd.read_csv(results / "csv" / f"error-by-{agg_by}.csv")
    mdi_error["log10(error)"] = np.log10(mdi_error["error"])

    g = sns.catplot(
        data=mdi_error,
        x=agg_by,
        y="log10(error)",
        row="subproblem",
        col="subproblem_id",
        kind="box",
    )
    g.savefig(results / "plots" / f"error-by-{agg_by}.png")


@click.command()
@click.option("--results", type=click.Path(path_type=Path), default="results")
@click.option("--agg_by", type=str, required=True)
def cli(results, agg_by):
    visualize(results, agg_by)


if __name__ == "__main__":
    cli()
