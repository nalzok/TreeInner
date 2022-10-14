from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import click


standard_value = {
    "eta": 1e-2,
    "max_depth": 4,
    "min_child_weight": 1,
    "num_boost_round": 400,
    "reg_lambda": 1,
}


def visualize(results: Path, data_root: Path, agg_by: str, skip_forest: bool):
    sns.set_theme(style="white", rc={"savefig.dpi": 300})
    log_xscale = {"eta", "min_child_weight", "reg_lambda"}

    oracle_auc = pd.read_csv(results / "csv" / f"auc-by-{agg_by}-{data_root}.csv")
    oracle_auc = oracle_auc.rename(columns={
        "distribution": "Dataset",
        "subproblem": "Task",
        "gfa": "GFA",
        "ifa": "IFA",
        "domain": "Domain",
        "auc": "AUC",
    })
    oracle_auc["Task"] = oracle_auc["Task"].str.capitalize()
    oracle_auc["IFA"] = oracle_auc["IFA"].str.replace("SHAP", "TreeSHAP")
    oracle_auc["Domain"] = oracle_auc["Domain"].str.replace("in", "Train")
    oracle_auc["Domain"] = oracle_auc["Domain"].str.replace("out", "Valid")
    if skip_forest:
        oracle_auc = oracle_auc[oracle_auc["GFA"] != "ForestInner"]
    oracle_auc["Domain (IFA)"] = oracle_auc["Domain"] + " (" + oracle_auc["IFA"] + ")"

    by = ["Dataset", "Task", agg_by, "GFA", "Domain", "IFA"]
    mask = oracle_auc[agg_by] == standard_value[agg_by]
    auc_mean_std = oracle_auc[mask].groupby(by).agg(
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        risk_mean=("risk_valid", "mean"),
        risk_std=("risk_valid", "std"),
    )
    auc_mean_std = auc_mean_std.reset_index().drop(columns=agg_by)
    auc_mean_std = auc_mean_std.set_index(["Dataset", "Task", "GFA", "Domain", "IFA"])
    print(auc_mean_std.to_latex(
        float_format="%.4f",
        longtable=True,
        multirow=True,
        caption="AUC Scores for Model Trained with Standard Hyperparameters",
        label="table:standard-auc",
    ))

    suffix = str(data_root)
    if not skip_forest:
        suffix = suffix + "-supplementary"

    for metric in ("AUC", "score_noisy", "score_signal"):
        fig_name = results / "svg" / agg_by / f"{metric}-{suffix}.svg"
        fig_name.parent.mkdir(parents=True, exist_ok=True)
        print("Creating", fig_name)
        facet_kws = {
            "sharey": False,
            "legend_out": True,
        }
        if metric != "AUC":
            oracle_auc[metric] /= oracle_auc["score_norm"]

        g = sns.relplot(
            data=oracle_auc,
            x=agg_by,
            y=metric,
            row="Dataset",
            col="Task",
            kind="line",
            style="Domain (IFA)",
            hue="GFA",
            markers={
                "Train (PreDecomp)": "o",
                "Valid (PreDecomp)": "o",
                "Train (TreeSHAP)": "X",
                "Valid (TreeSHAP)": "X",
                "Train (Permutation)": "*",
                "Valid (Permutation)": "*",
            },
            dashes={
                "Train (PreDecomp)": (2, 1),
                "Valid (PreDecomp)": "",
                "Train (TreeSHAP)": (2, 1),
                "Valid (TreeSHAP)": "",
                "Train (Permutation)": (2, 1),
                "Valid (Permutation)": "",
            },
            estimator="mean",
            errorbar=('se', 1),
            err_style="bars",
            facet_kws=facet_kws,
        )
        if agg_by in log_xscale:
            g.set(xscale="log")

        handles, labels = g.axes[0][0].get_legend_handles_labels()
        g._legend.remove()
        g.fig.legend(handles, labels, ncol=3, loc='upper center',
                     bbox_to_anchor=(0.5, 0), frameon=False,
                     markerscale=1.5, fontsize='x-large')

        g.savefig(fig_name)

    fig_name = results / "svg" / agg_by / f"risk-{suffix}.svg"
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    print("Creating", fig_name)
    facet_kws = {}
    facet_kws["sharey"] = False

    g = sns.relplot(
        data=oracle_auc,
        x=agg_by,
        y="risk_valid",
        row="Dataset",
        col="Task",
        kind="line",
        estimator="mean",
        errorbar=('se', 1),
        err_style="bars",
        facet_kws=facet_kws,
    )
    if agg_by in log_xscale:
        g.set(xscale="log")
    g.savefig(fig_name)

    mdi_error = pd.read_csv(results / "csv" / f"error-by-{agg_by}-{data_root}.csv")
    mdi_error["log10(rel_error)"] = np.log10(mdi_error["rel_error"])

    fig_name = results / "svg" / agg_by / f"error-{suffix}.svg"
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    print("Creating", fig_name)
    facet_kws = {
        "sharex": False,
        "sharey": False,
    }
    g = sns.catplot(
        data=mdi_error,
        x=agg_by,
        y="log10(rel_error)",
        row="distribution",
        col="subproblem",
        kind="box",
        hue=agg_by,
        palette=sns.color_palette('Blues', n_colors=1),
    )
    g.savefig(fig_name)


@click.command()
@click.option("--results", type=click.Path(path_type=Path), default="final_results")
@click.option(
    "--data_root", type=click.Path(path_type=Path), default="04_aggregate_2019"
)
@click.option("--agg_by", type=str, required=True)
@click.option("--skip_forest", type=bool, required=False)
def cli(results: Path, data_root: Path, agg_by: str, skip_forest: bool):
    visualize(results, data_root, agg_by, skip_forest)


if __name__ == "__main__":
    cli()
