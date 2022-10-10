from typing import Dict, Tuple, Sequence, List
from numbers import Number
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score
from .importance import (
    train_boosters,
    evaluate_boosters,
    compute_contribution_gradient,
    feature_importance,
    permutation_importance,
    validate_total_gain,
)
from .visualize import visualize


def main(
    data_root: Path, grid: Dict[str, Tuple[Number, Sequence[Number]]], agg_by: str
):
    keys = ("eta", "max_depth", "min_child_weight", "num_boost_round", "reg_lambda")
    for name in keys:
        assert name in grid, f"{name} should be in param."
    assert agg_by in keys, "Must aggregate by a hyperparameter"

    oracle_auc_row = []
    mdi_error_row = []

    for distribution_id in (2, 1):
        for subproblem in ("classification", "regression"):
            print(f"Working on {subproblem}{distribution_id}")
            subdirectory = data_root / f"{subproblem}{distribution_id}"
            for dataset_id in trange(40, leave=False):
                experiment(
                    subdirectory,
                    grid.copy(),
                    agg_by,
                    subproblem,
                    distribution_id,
                    dataset_id,
                    oracle_auc_row,
                    mdi_error_row,
                )

    results = Path("results")
    (results / "csv").mkdir(parents=True, exist_ok=True)
    (results / "plots").mkdir(parents=True, exist_ok=True)

    oracle_auc = pd.DataFrame(oracle_auc_row)
    oracle_auc.to_csv(results / "csv" / f"auc-by-{agg_by}-{data_root}.csv")

    mdi_error = pd.DataFrame(mdi_error_row)
    mdi_error.to_csv(results / "csv" / f"error-by-{agg_by}-{data_root}.csv")

    visualize(results, data_root, agg_by)


def experiment(
    subdirectory: Path,
    grid: Dict[str, Tuple[Number, Sequence[Number]]],
    agg_by: str,
    subproblem: str,
    distribution_id: int,
    dataset_id: int,
    oracle_auc_row: List,
    mdi_error_row: List,
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

    signal = 1 - noisy.round(0).astype(int)
    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)

    objective = (
        "binary:logistic" if subproblem == "classification" else "reg:squarederror"
    )
    base_score = 0.5 if subproblem == "classification" else 0.0
    param = {
        "objective": objective,
        "base_score": base_score,
    }

    _, sweep = grid.pop(agg_by)
    for k, (default, _) in grid.items():
        param[k] = default

    for val in tqdm(sweep, leave=False):
        param[agg_by] = val
        num_boost_round = param.pop("num_boost_round")

        boosters = train_boosters(dtrain, num_boost_round, param)
        risk_train = evaluate_boosters(dtrain, boosters, num_boost_round, param)
        risk_valid = evaluate_boosters(dvalid, boosters, num_boost_round, param)

        distribution = "ChIP" if distribution_id == 1 else "Simulated"
        common = {
            "distribution": distribution,
            "subproblem": subproblem,
            "dataset_id": dataset_id,
            "risk_train": risk_train,
            "risk_valid": risk_valid,
            agg_by: val,
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

        total_gain = None
        for use_valid in (False, True):
            dimportance = dvalid if use_valid else dtrain
            domain = "out" if use_valid else "in"
            for ifa in ("PreDecomp", "SHAP"):
                contributions, gradient = compute_contribution_gradient(
                    dimportance, boosters, num_boost_round, param, ifa
                )
                for gfa in ("Abs", "Inner"):
                    score = feature_importance(
                        contributions,
                        gradient,
                        param,
                        gfa,
                    )
                    if use_valid is False and ifa == "PreDecomp" and gfa == "Inner":
                        total_gain = score

                    oracle_auc_row.append(
                        {
                            "gfa": gfa,
                            "ifa": ifa,
                            "domain": domain,
                            "auc_noisy": roc_auc_score(signal, score),
                            **common,
                        }
                    )

        assert total_gain is not None, "Remember to calculate total gain estimation"

        rel_error = validate_total_gain(total_gain, dtrain, num_boost_round, param)
        mdi_error_row.append(
            {
                "rel_error": rel_error,
                **common,
            }
        )

        param["num_boost_round"] = num_boost_round


if __name__ == "__main__":
    assert xgb.__version__ == "1.6.2-dev", "A custom fork of XGBoost is required."

    np.random.seed(42)

    data_roots = (Path("04_aggregate_2019"), Path("04_aggregate_2022"))

    grid = {
        # name: (default, sweep)
        "eta": (0.3, (0.01, 0.03, 0.1, 0.3, 1)),
        "max_depth": (6, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        "min_child_weight": (1, (0, 0.5, 1, 2, 4, 8)),
        "num_boost_round": (400, (200, 400, 600, 800, 1000)),
        "reg_lambda": (1, (0.1, 1, 5, 10, 50, 100)),
    }
    agg_by = "num_boost_round"

    for data_root in data_roots:
        main(data_root, grid, agg_by)
