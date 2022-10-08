from pathlib import Path
from timeit import default_timer as timer

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score
import seaborn as sns
from .importance import (
    train_boosters,
    feature_importance,
    permutation_importance,
    validate_total_gain,
)


def main(data_root, param, num_boost_rounds):
    for name in ["max_depth", "eta", "reg_lambda"]:
        assert name in param, "{} should be in param.".format(name)

    param_str = "+".join(k + "=" + str(v) for k, v in param.items()).replace(".", "p")

    oracle_auc_row = []
    mdi_error_row = []

    for subproblem in ("classification", "regression"):
        param["objective"] = (
            "binary:logistic" if subproblem == "classification" else "reg:squarederror"
        )
        param["base_score"] = 0.5 if subproblem == "classification" else 0.0

        for subproblem_id in (1, 2):
            print(f"Working on {subproblem}{subproblem_id}")
            subdirectory = data_root / f"{subproblem}{subproblem_id}"
            for dataset_id in trange(40, leave=False):
                experiment(
                    subdirectory,
                    param,
                    num_boost_rounds,
                    subproblem,
                    subproblem_id,
                    dataset_id,
                    oracle_auc_row,
                    mdi_error_row,
                )

    results = Path("results")
    (results / "csv").mkdir(parents=True, exist_ok=True)
    (results / "plots").mkdir(parents=True, exist_ok=True)

    oracle_auc = pd.DataFrame(oracle_auc_row)
    oracle_auc.to_csv(results / "csv" / f"oracle-auc+{param_str}.csv")

    mdi_error = pd.DataFrame(mdi_error_row)
    mdi_error.to_csv(results / "csv" / f"mdi-error+{param_str}.csv")

    visualize(results, param_str)


def experiment(
    subdirectory,
    param,
    num_boost_rounds,
    subproblem,
    subproblem_id,
    dataset_id,
    oracle_auc_row,
    mdi_error_row,
):
    X_train = pd.read_csv(subdirectory / f"permuted{dataset_id}_X_train.csv", header=None)
    Y_train = pd.read_csv(subdirectory / f"permuted{dataset_id}_y_train.csv", header=None)
    X_valid = pd.read_csv(subdirectory / f"permuted{dataset_id}_X_test.csv", header=None)
    Y_valid = pd.read_csv(subdirectory / f"permuted{dataset_id}_y_test.csv", header=None)
    noisy = pd.read_csv(subdirectory / f"permuted{dataset_id}_noisy_features.csv", header=None)

    if subproblem == "regression":
        Y_train[0] -= Y_train[0].mean()
        Y_valid[0] -= Y_valid[0].mean()

    signal = -noisy.round(0).astype(int)
    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)

    boosters = train_boosters(dtrain, max(num_boost_rounds), param)
    for num_boost_round in tqdm(num_boost_rounds, leave=False):
        common = {
            "subproblem": subproblem,
            "subproblem_id": subproblem_id,
            "dataset_id": dataset_id,
            "num_boost_round": num_boost_round,
        }

        start = timer()
        score = permutation_importance(
            boosters, num_boost_round, X_valid, Y_valid, param, 5
        )
        end = timer()
        oracle_auc_row.append(
            {
                "method": "Permutation",
                "auc_noisy": roc_auc_score(signal, score),
                "elapsed": end - start,
                **common,
            }
        )

        total_gain = None
        for correlation in ("Abs", "Inner"):
            for oob in (False, True):
                dimportance = dvalid if oob else dtrain
                domain = "in" if oob else "out"
                for algo in ("PreDecomp", "SHAP"):
                    start = timer()
                    score = feature_importance(
                        dimportance, boosters, num_boost_round, param, correlation, algo
                    )
                    end = timer()
                    if (
                        correlation == "Inner"
                        and oob is False
                        and algo == "PreDecomp"
                    ):
                        total_gain = score

                    oracle_auc_row.append(
                        {
                            "method": f"{correlation}-{domain}-{algo}",
                            "auc_noisy": roc_auc_score(signal, score),
                            "elapsed": end - start,
                            **common,
                        }
                    )
        assert total_gain is not None, "Remember to calculate total gain estimation"

        error = validate_total_gain(total_gain, dtrain, param, num_boost_round)
        mdi_error_row.append(
            {
                "error": error,
                **common,
            }
        )


def visualize(results, param_str):
    oracle_auc = pd.read_csv(results / "csv" / f"oracle-auc+{param_str}.csv")
    sns_plot = sns.catplot(
        x="num_boost_round",
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

    oracle_auc["log10(elapsed)"] = np.log10(oracle_auc["elapsed"])
    sns_plot = sns.catplot(
        x="num_boost_round",
        y="log10(elapsed)",
        col="subproblem",
        row="subproblem_id",
        hue="method",
        kind="box",
        data=oracle_auc,
        height=16,
        aspect=2,
    )
    sns_plot.savefig(results / "plots" / f"oracle-elapsed+{param_str}.png")

    mdi_error = pd.read_csv(results / "csv" / f"mdi-error+{param_str}.csv")
    mdi_error["log10(error)"] = np.log10(mdi_error["error"])
    sns_plot = sns.catplot(
        x="num_boost_round",
        y="log10(error)",
        col="subproblem",
        row="subproblem_id",
        kind="box",
        palette=sns.color_palette("Blues", n_colors=len(num_boost_rounds)),
        data=mdi_error,
    )
    sns_plot.savefig(results / "plots" / f"mdi-error+{param_str}.png")


if __name__ == "__main__":
    assert (
        xgb.__version__ == "1.6.2-dev"
    ), "A custom fork of XGBoost is required. See https://github.com/nalzok/xgboost/tree/release_1.6.0"

    np.random.seed(42)
    sns.set_theme(style="whitegrid", rc={"savefig.dpi": 300})

    data_root = Path("04_aggregate")

    param = {
        "max_depth": 5,
        "eta": 0.1,
        "reg_lambda": 1,
    }

    num_boost_rounds = (250, 500, 1000)

    main(data_root, param, num_boost_rounds)
