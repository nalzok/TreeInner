import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    permutation_importance,
    feature_importance,
    evaluate_total_gain,
    generalization_error,
)

num_boost_rounds = (1, 2, 4, 8, 16, 32, 64, 128)

param = {
    "max_depth": 4,
    "eta": 0.1,
    "base_score": 0.5,
    "reg_lambda": 1,
}

DATA_PATH = "04_aggregate"

def experiment(
    subproblem, subproblem_id, i, oracle_auc_row, mdi_error_row, error_test_row
):
    data_path = f"{DATA_PATH}/{subproblem}{subproblem_id}"

    Y_full = pd.read_csv(f"{data_path}/permuted{i}_y_train.csv", header=None)
    X_full = pd.read_csv(f"{data_path}/permuted{i}_X_train.csv", header=None)
    nrow = len(X_full.index)
    indices = np.random.permutation(nrow)
    thredhold = int(nrow * 0.5)
    training_idx, validation_idx = indices[:thredhold], indices[thredhold:]
    Y_train, Y_valid = Y_full.iloc[training_idx], Y_full.iloc[validation_idx]
    X_train, X_valid = X_full.iloc[training_idx], X_full.iloc[validation_idx]

    Y_test = pd.read_csv(f"{data_path}/permuted{i}_y_test.csv", header=None)
    X_test = pd.read_csv(f"{data_path}/permuted{i}_X_test.csv", header=None)

    noisy = (
        pd.read_csv(f"{data_path}/permuted{i}_noisy_features.csv", header=None)
        .round(0)
        .astype(int)
    )

    for num_boost_round in tqdm(num_boost_rounds, leave=False):
        (
            total_gain_oob_saabas,
            coef_saabas,
            total_gain_oob_saabas_elapsed,
        ) = feature_importance(
            Y_train, X_train, Y_valid, X_valid, param, num_boost_round, use_saabas=True
        )
        common = {
            "subproblem": subproblem,
            "subproblem_id": subproblem_id,
            "dataset_id": i,
            "num_boost_round": num_boost_round,
        }

        oracle_auc_row.append(
            {
                "method": "TotalGain-oob (Saabas)",
                "auc_noisy": metrics.roc_auc_score(-noisy, total_gain_oob_saabas),
                "elapsed": total_gain_oob_saabas_elapsed,
                **common,
            }
        )

        total_gain_oob_shap, coef_shap, total_gain_oob_shap_elapsed = feature_importance(
            Y_train, X_train, Y_valid, X_valid, param, num_boost_round, use_saabas=False
        )
        oracle_auc_row.append(
            {
                "method": "TotalGain-oob (Shap)",
                "auc_noisy": metrics.roc_auc_score(-noisy, total_gain_oob_shap),
                "elapsed": total_gain_oob_shap_elapsed,
                **common,
            }
        )

        total_gain, _, total_gain_elapsed = feature_importance(
            Y_full, X_full, Y_full, X_full, param, num_boost_round, use_saabas=True
        )
        oracle_auc_row.append(
            {
                "method": "TotalGain",
                "auc_noisy": metrics.roc_auc_score(-noisy, total_gain),
                "elapsed": total_gain_elapsed,
                **common,
            }
        )

        total_gain_shap, _, total_gain_shap_elapsed = feature_importance(
            Y_full, X_full, Y_full, X_full, param, num_boost_round, use_saabas=False
        )
        oracle_auc_row.append(
            {
                "method": "TotalGain (Shap)",
                "auc_noisy": metrics.roc_auc_score(-noisy, total_gain_shap),
                "elapsed": total_gain_shap_elapsed,
                **common,
            }
        )

        permuation, permutation_elapsed = permutation_importance(
            Y_train, X_train, Y_valid, X_valid, param, num_boost_round, perm_round=5
        )
        oracle_auc_row.append(
            {
                "method": "Permutation (perm_round = 5)",
                "auc_noisy": metrics.roc_auc_score(-noisy, permuation),
                "elapsed": permutation_elapsed,
                **common,
            }
        )

        total_gain_error = evaluate_total_gain(
            total_gain, Y_full, X_full, param, num_boost_round
        )

        mdi_error_row.append(
            {
                "error": total_gain_error,
                **common,
            }
        )

        error_test_vanilla = generalization_error(
            Y_full, X_full, Y_test, X_test, param, num_boost_round,
        )
        error_test_row.append(
            {
                "method": "Vanilla",
                "error_test": error_test_vanilla,
                **common,
            }
        )

        # maybe the first two parameters can be Y_full, X_full?
        error_test_lasso_saabas = generalization_error(
            Y_train,
            X_train,
            Y_test,
            X_test,
            param,
            num_boost_round,
            coef=coef_saabas,
            use_saabas=True,
        )
        error_test_row.append(
            {
                "method": "LASSO (Saabas)",
                "error_test": error_test_lasso_saabas,
                **common,
            }
        )

        error_test_lasso_shap = generalization_error(
            Y_train,
            X_train,
            Y_test,
            X_test,
            param,
            num_boost_round,
            coef=coef_shap,
            use_saabas=False,
        )
        error_test_row.append(
            {
                "method": "LASSO (Shap)",
                "error_test": error_test_lasso_shap,
                **common,
            }
        )


def visualize(oracle_auc, mdi_error, error_test):
    oracle_auc = pd.read_csv(f"results/oracle-auc+{param_str}.csv")
    sns_plot = sns.catplot(
        x="num_boost_round",
        y="auc_noisy",
        col="subproblem",
        row="subproblem_id",
        hue="method",
        kind="box",
        data=oracle_auc,
    )
    sns_plot.savefig(f"results/oracle-auc+{param_str}.png")

    oracle_auc["log10(elapsed)"] = np.log10(oracle_auc["elapsed"])
    sns_plot = sns.catplot(
        x="num_boost_round",
        y="log10(elapsed)",
        col="subproblem",
        row="subproblem_id",
        hue="method",
        kind="box",
        data=oracle_auc,
    )
    sns_plot.savefig(f"results/oracle-elapsed+{param_str}.png")

    mdi_error = pd.read_csv(f"results/mdi-error+{param_str}.csv")
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

    sns_plot.savefig(f"results/mdi-error+{param_str}.png")

    error_test = pd.read_csv(f"results/error-test+{param_str}.csv")
    sns_plot = sns.catplot(
        x="num_boost_round",
        y="error_test",
        col="subproblem",
        row="subproblem_id",
        hue="method",
        kind="box",
        data=error_test,
    )
    sns_plot.savefig(f"results/error-test+{param_str}.png")

    for subproblem in ("classification", "regression"):
        for subproblem_id in (1, 2):
            error_test_quadrant = error_test[
                (error_test["subproblem"] == subproblem)
                & (error_test["subproblem_id"] == subproblem_id)
            ]
            ax = sns.boxplot(
                x="num_boost_round", y="error_test", hue="method", data=error_test_quadrant
            )
            ax.figure.savefig(f"results/error-test+{param_str}+{subproblem}{subproblem_id}.png")
            plt.close(ax.figure)


if __name__ == "__main__":
    assert xgb.__version__.endswith(
        "-lambda"
    ), "A custom fork of XGBoost is required. See https://github.com/nalzok/xgboost"

    sns.set(style="whitegrid")
    np.random.seed(42)

    param_str = "+".join(k + "=" + str(v) for k, v in param.items()).replace('.', 'p')

    oracle_auc_row = []
    mdi_error_row = []
    error_test_row = []

    for subproblem in ("classification", "regression"):
        param["objective"] = (
            "binary:logistic" if subproblem == "classification" else "reg:squarederror"
        )
        for subproblem_id in (1, 2):
            for i in trange(40):
                experiment(
                    subproblem,
                    subproblem_id,
                    i,
                    oracle_auc_row,
                    mdi_error_row,
                    error_test_row,
                )

    oracle_auc = pd.DataFrame(oracle_auc_row)
    oracle_auc.to_csv(f"results/oracle-auc+{param_str}.csv")

    mdi_error = pd.DataFrame(mdi_error_row)
    mdi_error.to_csv(f"results/mdi-error+{param_str}.csv")

    error_test = pd.DataFrame(error_test_row)
    error_test.to_csv(f"results/error-test+{param_str}.csv")

    visualize(oracle_auc, mdi_error, error_test)

