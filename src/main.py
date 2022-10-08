from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score
import seaborn as sns
from .importance import (
    train_boosters,
    compute_contribution_gradient,
    feature_importance,
    permutation_importance,
    validate_total_gain,
)


def main(data_root, num_boost_round, param, min_child_weight_list):
    for name in ["eta", "reg_lambda"]:
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
                    num_boost_round,
                    param,
                    min_child_weight_list,
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
    num_boost_round,
    param,
    min_child_weight_list,
    subproblem,
    subproblem_id,
    dataset_id,
    oracle_auc_row,
    mdi_error_row,
):
    X_train = pd.read_csv(
        subdirectory / f"permuted{dataset_id}_X_train.csv", header=None
    )
    Y_train = pd.read_csv(
        subdirectory / f"permuted{dataset_id}_y_train.csv", header=None
    )
    X_valid = pd.read_csv(
        subdirectory / f"permuted{dataset_id}_X_test.csv", header=None
    )
    Y_valid = pd.read_csv(
        subdirectory / f"permuted{dataset_id}_y_test.csv", header=None
    )
    noisy = pd.read_csv(
        subdirectory / f"permuted{dataset_id}_noisy_features.csv", header=None
    )

    signal = -noisy.round(0).astype(int)
    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)

    for min_child_weight in tqdm(min_child_weight_list, leave=False):
        param["min_child_weight"] = min_child_weight
        boosters = train_boosters(dtrain, num_boost_round, param)

        common = {
            "subproblem": subproblem,
            "subproblem_id": subproblem_id,
            "dataset_id": dataset_id,
            "min_child_weight": min_child_weight,
        }

        # score = permutation_importance(
        #     boosters, num_boost_round, X_valid, Y_valid, param, 5
        # )
        # oracle_auc_row.append(
        #     {
        #         "method": "Permutation",
        #         "auc_noisy": roc_auc_score(signal, score),
        #         **common,
        #     }
        # )

        use_valid_list = (False, True)
        ifa_list = ("PreDecomp", "SHAP")
        gfa_list = ("Abs", "Inner")
        result = {}

        total_gain = None
        for use_valid in use_valid_list:
            dimportance = dvalid if use_valid else dtrain
            for ifa in ifa_list:
                contributions, gradient = compute_contribution_gradient(
                    dimportance, boosters, num_boost_round, param, ifa
                )
                for gfa in gfa_list:
                    score = feature_importance(
                        contributions,
                        gradient,
                        param,
                        gfa,
                    )
                    if use_valid is False and ifa == "PreDecomp" and gfa == "Inner":
                        total_gain = score
                    result[(use_valid, ifa, gfa)] = score
        assert total_gain is not None, "Remember to calculate total gain estimation"

        # reorder experiment results
        for gfa in gfa_list:
            for ifa in ifa_list:
                for use_valid in use_valid_list:
                    domain = "in" if use_valid else "out"
                    score = result[(use_valid, ifa, gfa)]
                    oracle_auc_row.append(
                        {
                            "method": f"{gfa}-{ifa}-{domain}",
                            "auc_noisy": roc_auc_score(signal, score),
                            **common,
                        }
                    )

        error = validate_total_gain(total_gain, dtrain, num_boost_round, param)
        mdi_error_row.append(
            {
                "error": error,
                **common,
            }
        )


def visualize(results, param_str):
    oracle_auc = pd.read_csv(results / "csv" / f"oracle-auc+{param_str}.csv")
    sns_plot = sns.catplot(
        x="min_child_weight",
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

    mdi_error = pd.read_csv(results / "csv" / f"mdi-error+{param_str}.csv")
    mdi_error["log10(error)"] = np.log10(mdi_error["error"])
    sns_plot = sns.catplot(
        x="min_child_weight",
        y="log10(error)",
        col="subproblem",
        row="subproblem_id",
        kind="box",
        data=mdi_error,
    )
    sns_plot.savefig(results / "plots" / f"mdi-error+{param_str}.png")


if __name__ == "__main__":
    assert (
        xgb.__version__ == "1.6.2-dev"
    ), "A custom fork of XGBoost is required."

    np.random.seed(42)
    sns.set_theme(style="whitegrid", rc={"savefig.dpi": 300})

    data_root = Path("04_aggregate")

    num_boost_round = 500
    param = {
        "eta": 0.1,
        "reg_lambda": 1,
    }
    min_child_weight_list = (1, 100)

    main(data_root, num_boost_round, param, min_child_weight_list)
