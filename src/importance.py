from typing import Dict, List, Tuple
from numbers import Number

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb


margin2pred = {
    "reg:squarederror": lambda margins: margins,
    "reg:linear": lambda margins: margins,
    "binary:logitraw": expit,
    "reg:logistic": expit,
    "binary:logistic": expit,
}


def train_boosters(
    dtrain: xgb.DMatrix,
    num_boost_round: int,
    param: Dict,
) -> List[xgb.Booster]:
    # reset the margin of dtrain
    dtrain.set_base_margin([])

    boosters = []
    for _ in range(num_boost_round):
        # train a new tree
        bst = xgb.train(param, dtrain, 1, verbose_eval=False)
        boosters.append(bst)

        # output the prediction decomposition using all the available trees
        ptrain_tree = bst.predict(dtrain, output_margin=True)
        # set the margin to incorporate all the previous trees' prediction
        dtrain.set_base_margin(ptrain_tree)

    return boosters


def evaluate_boosters(
    dimportance: xgb.DMatrix,
    boosters: List[xgb.Booster],
    num_boost_round: int,
    param: Dict,
) -> Number:
    assert (
        len(boosters) == num_boost_round
    ), f"Ensemble size mismatch: {len(boosters)} != {num_boost_round}"

    dimportance.set_base_margin([])

    # calculate baseline prediction accuracy
    base_margin = np.full(dimportance.num_row(), param["base_score"])
    for bst in boosters[:num_boost_round]:
        base_margin = bst.predict(dimportance, output_margin=True)
        dimportance.set_base_margin(base_margin)
    trans = margin2pred[param["objective"]]
    predictions = trans(base_margin)

    if param["objective"] in ("reg:squarederror", "reg:linear"):
        risk = -mean_squared_error(dimportance.get_label(), predictions)
    else:
        risk = roc_auc_score(dimportance.get_label(), predictions)

    return risk


def compute_contribution_gradient(
    dimportance: xgb.DMatrix,
    boosters: List[xgb.Booster],
    num_boost_round: int,
    param: Dict,
    ifa: str,
) -> Tuple[np.ndarray, np.ndarray]:
    assert (
        len(boosters) == num_boost_round
    ), f"Ensemble size mismatch: {len(boosters)} != {num_boost_round}"

    # reset the margin of dimportance
    dimportance.set_base_margin([])

    # store the feature contribution of each tree
    contributions_by_tree = np.empty(
        (num_boost_round, dimportance.num_row(), dimportance.num_col() + 1),
        dtype=np.float32,
    )

    # store the (negative) gradient for each tree
    gradient_by_tree = np.zeros(
        (num_boost_round, dimportance.num_row()), dtype=np.float32
    )
    gradient_by_tree[0, :] = dimportance.get_label() - param["base_score"]

    base_margin = np.full(dimportance.num_row(), param["base_score"])
    for t, bst in enumerate(boosters[:num_boost_round]):
        # compute the contribution_by_tree
        if ifa == "PreDecomp":
            pimportance_tree = bst.predict(
                dimportance,
                pred_contribs=True,
                approx_contribs=True,
                reg_lambda=param["reg_lambda"],
            )
        elif ifa == "ApproxSHAP":
            pimportance_tree = bst.predict(
                dimportance,
                pred_contribs=True,
                approx_contribs=True,
                reg_lambda=0,
            )
        elif ifa == "SHAP":
            pimportance_tree = bst.predict(dimportance, pred_contribs=True)
        else:
            raise ValueError(f"Unknown IFA {ifa}")

        contributions_by_tree[t, :, :-1] = pimportance_tree[:, :-1]
        contributions_by_tree[t, :, -1] = pimportance_tree[:, -1] - base_margin
        base_margin = np.sum(pimportance_tree, axis=-1)

        trans = margin2pred[param["objective"]]
        gradient_by_tree[t, :] = dimportance.get_label() - trans(base_margin)

        # set the margin to incorporate all the previous trees' prediction
        dimportance.set_base_margin(base_margin)

    return contributions_by_tree, gradient_by_tree


def feature_importance(
    contributions_by_tree: np.ndarray,
    gradient_by_tree: np.ndarray,
    param: Dict,
    gfa: str,
) -> np.ndarray:
    if gfa == "Inner":
        gradient_by_tree = gradient_by_tree[:, :, np.newaxis]
        MDI = np.sum(contributions_by_tree * gradient_by_tree, axis=(0, 1))
        MDI = MDI[:-1] / param["eta"]
    elif gfa == "Abs":
        # ignore the per-tree bias
        MDI = np.sum(np.abs(contributions_by_tree[:, :, :-1]), axis=(0, 1))
    else:
        raise ValueError(f"Unknown GFA {gfa}")

    return MDI


def permutation_importance(
    boosters: List[xgb.Booster],
    num_boost_round: int,
    X_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    param: Dict,
    perm_round: int,
) -> np.ndarray:
    assert (
        len(boosters) == num_boost_round
    ), f"Ensemble size mismatch: {len(boosters)} != {num_boost_round}"

    # build DMatrix from data frames
    dimportance = xgb.DMatrix(X_valid, Y_valid, silent=True)

    # calculate baseline prediction accuracy
    base_margin = np.full(dimportance.num_row(), param["base_score"])
    for bst in boosters[:num_boost_round]:
        base_margin = bst.predict(dimportance, output_margin=True)
        dimportance.set_base_margin(base_margin)
    trans = margin2pred[param["objective"]]
    predictions = trans(base_margin)

    if param["objective"] in ("reg:squarederror", "reg:linear"):
        baseline = -mean_squared_error(dimportance.get_label(), predictions)
    else:
        baseline = roc_auc_score(dimportance.get_label(), predictions)

    importances = np.zeros((len(X_valid.columns),), dtype=np.float32)
    for idx, feature in enumerate(X_valid.columns):
        loss_sum = 0
        for _ in range(perm_round):
            # permute row and re-calculate prediction accuracy
            save = X_valid[feature].copy()
            X_valid[feature] = np.random.permutation(X_valid[feature])
            dimportance = xgb.DMatrix(X_valid, Y_valid, silent=True)
            X_valid[feature] = save

            margin = np.full(dimportance.num_row(), param["base_score"])
            for bst in boosters[:num_boost_round]:
                margin = bst.predict(dimportance, output_margin=True)
                dimportance.set_base_margin(margin)
            trans = margin2pred[param["objective"]]
            predictions = trans(margin)

            if param["objective"] in ("reg:squarederror", "reg:linear"):
                loss_sum += -mean_squared_error(dimportance.get_label(), predictions)
            else:
                loss_sum += roc_auc_score(dimportance.get_label(), predictions)

        importances[idx] = baseline - loss_sum / perm_round

    return importances


def validate_total_gain(
    candidate: np.ndarray, dtrain: xgb.DMatrix, num_boost_round: int, param: Dict
) -> float:
    dtrain.set_base_margin([])

    bst_all_in_one = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)
    assert bst_all_in_one.feature_names is not None
    total_gain_dict = bst_all_in_one.get_score(importance_type="total_gain")
    total_gain = np.array(
        [total_gain_dict.get(f, 0) for f in bst_all_in_one.feature_names]
    ).astype("float64")

    total_gain /= np.sum(total_gain)
    candidate /= np.sum(candidate)

    return np.max(np.abs(total_gain - candidate))
