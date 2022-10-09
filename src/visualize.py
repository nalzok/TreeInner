from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import click


def visualize(results, param_str):
    sns.set_theme(style="whitegrid", rc={"savefig.dpi": 300})

    oracle_auc = pd.read_csv(results / "csv" / f"oracle-auc+{param_str}.csv")
    mdi_error = pd.read_csv(results / "csv" / f"mdi-error+{param_str}.csv")
    mdi_error["log10(error)"] = np.log10(mdi_error["error"])

    by = ["subproblem", "subproblem_id", "max_depth", "method"]
    auc_mean_std = oracle_auc.groupby(by).agg(
        auc_mean=("auc_noisy", "mean"), auc_std=("auc_noisy", "std")
    )
    print(auc_mean_std.to_latex())

    sns_plot = sns.catplot(
        x="max_depth",
        y="auc_noisy",
        col="subproblem",
        row="subproblem_id",
        hue="method",
        kind="box",
        data=oracle_auc,
        height=16,
        aspect=2,
    )
    sns_plot.savefig(results / "plots" / f"oracle-auc+{param_str}.png")

    sns_plot = sns.catplot(
        x="max_depth",
        y="log10(error)",
        col="subproblem",
        row="subproblem_id",
        kind="box",
        data=mdi_error,
    )
    sns_plot.savefig(results / "plots" / f"mdi-error+{param_str}.png")


@click.command()
@click.option("--results", type=click.Path(path_type=Path), default="results")
@click.option("--config_name", type=str, required=True)
def cli(results, config_name):
    visualize(results, config_name)


if __name__ == "__main__":
    cli()
