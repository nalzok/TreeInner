from typing import Dict, Tuple
from warnings import catch_warnings, simplefilter
from time import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb


transformation = {
    "reg:squarederror": lambda margins: margins,
    "reg:linear": lambda margins: margins,
    "binary:logitraw": expit,
    "reg:logistic": expit,
    "binary:logistic": expit,
}


def feature_importance(
    dtrain: xgb.DMatrix,
    dimportance: xgb.DMatrix,
    param: Dict,
    num_boost_round: int,
    correlation: str,
    algo: str,
) -> Tuple[np.ndarray, float]:
    time_begin = time()

    # reset the margin of dtrain & dimportance
    dtrain.set_base_margin([])
    dimportance.set_base_margin([])

    # train a boosting forest with DTRAIN, use it to make prediction for dimportance,
    # and decompose the result into feature contributions
    contributions_by_tree = _compute_contribution(
        dtrain, dimportance, param, num_boost_round, algo
    )

    # compute gradient
    gradient_by_tree = _compute_gradient(
        dimportance,
        contributions_by_tree,
        param["objective"],
        param["base_score"],
        num_boost_round,
    )
    gradient_by_tree = gradient_by_tree[:, :, np.newaxis]

    if correlation == "Covariance":
        MDI = np.sum(contributions_by_tree * gradient_by_tree, axis=(0, 1))
        MDI = MDI[:-1] / param["eta"]
    elif correlation == "Pearson":
        MDI = np.zeros(contributions_by_tree.shape[-1] - 1)
        with catch_warnings():
            simplefilter("ignore", ConstantInputWarning)
            for k in range(MDI.size):
                for t in range(gradient_by_tree.shape[0]):
                    y_true = gradient_by_tree[t, :, 0]
                    y_score = contributions_by_tree[t, :, k]
                    corr = pearsonr(y_true, y_score)
                    if not np.isnan(corr.correlation):
                        MDI[k] += corr.correlation
    elif correlation == "Spearman":
        MDI = np.zeros(contributions_by_tree.shape[-1] - 1)
        with catch_warnings():
            simplefilter("ignore", ConstantInputWarning)
            for k in range(MDI.size):
                for t in range(gradient_by_tree.shape[0]):
                    y_true = gradient_by_tree[t, :, 0]
                    y_score = contributions_by_tree[t, :, k]
                    corr = spearmanr(y_true, y_score)
                    if not np.isnan(corr.correlation):
                        MDI[k] += corr.correlation
    elif correlation == "AbsoluteValue":
        # ignore the per-tree bias
        MDI = np.sum(np.abs(contributions_by_tree[:, :, :-1]), axis=(0, 1))
    else:
        raise ValueError(f"Unknown correlation metric {correlation}")

    return MDI, time() - time_begin


def _compute_contribution(
    dtrain: xgb.DMatrix,
    dimportance: xgb.DMatrix,
    param: Dict,
    num_boost_round: int,
    algo: str,
) -> np.ndarray:
    # store the feature contribution of each tree
    contributions_by_tree = np.zeros(
        (num_boost_round, dimportance.num_row(), dimportance.num_col() + 1),
        dtype=np.float32,
    )

    for t in range(num_boost_round):
        # train a new tree
        bst = xgb.train(param, dtrain, 1, verbose_eval=False)

        # compute the contribution_by_tree
        if algo == "Saabas":
            pvalid_tree = bst.predict(
                dimportance,
                pred_contribs=True,
                approx_contribs=True,
                reg_lambda=param["reg_lambda"],
            )
        elif algo == "ApproxSHAP":
            pvalid_tree = bst.predict(
                dimportance,
                pred_contribs=True,
                approx_contribs=True,
                reg_lambda=0,
            )
        elif algo == "SHAP":
            pvalid_tree = bst.predict(dimportance, pred_contribs=True)
        else:
            raise ValueError(f"Unknown algorithm {algo}")

        contributions_by_tree[t, :, :-1] = pvalid_tree[:, :-1]
        if t == 0:
            contributions_by_tree[t, :, -1] = pvalid_tree[:, -1]
        else:
            contributions_by_tree[t, :, -1] = pvalid_tree[:, -1] - np.sum(
                contributions_by_tree[:t, :, :], axis=(0, 2)
            )

        # output the prediction decomposition using all the available trees
        ptrain_tree = bst.predict(dtrain, output_margin=True)

        # reset the margin to incorporate all the previous trees' prediction
        dtrain.set_base_margin(ptrain_tree)
        dimportance.set_base_margin(np.sum(pvalid_tree, axis=1))

    return contributions_by_tree


def _compute_gradient(
    dimportance: xgb.DMatrix,
    contributions_by_tree: np.ndarray,
    objective: str,
    base_score: float,
    num_boost_round: int,
) -> np.ndarray:
    # compute gradient for each tree
    gradient_by_tree = np.zeros(
        (num_boost_round, dimportance.num_row()), dtype=np.float32
    )
    gradient_by_tree[0, :] = dimportance.get_label() - base_score

    # TODO: is this correct for classification?
    trans = transformation[objective]
    for t in range(1, num_boost_round):
        gradient_by_tree[t, :] = dimportance.get_label() - trans(
            np.sum(contributions_by_tree[:t, :, :], axis=(0, 2))
        )

    return gradient_by_tree


def permutation_importance(
    dtrain: xgb.DMatrix,
    X_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    param: Dict,
    num_boost_round: int,
    perm_round: int,
) -> Tuple[np.ndarray, float]:
    time_begin = time()

    # train a boosting forest with DTRAIN, which will be used to make prediction for dimportance
    bst = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)

    # build DMatrix from data frames
    dimportance = xgb.DMatrix(X_valid, Y_valid, silent=True)

    # calculate baseline prediction accuracy
    predictions = bst.predict(dimportance)
    if param["objective"] in ("reg:squarederror", "reg:linear"):
        baseline = mean_squared_error(dimportance.get_label(), predictions)
    else:
        baseline = 1 - roc_auc_score(dimportance.get_label(), predictions)

    importances = np.zeros((len(X_valid.columns),), dtype=np.float32)
    for idx, feature in enumerate(X_valid.columns):
        # permute row and re-calculate prediction accuracy
        loss_sum = 0
        for _ in range(perm_round):
            save = X_valid[feature].copy()
            X_valid[feature] = np.random.permutation(X_valid[feature])
            dimportance = xgb.DMatrix(X_valid, Y_valid, silent=True)
            X_valid[feature] = save

            predictions = bst.predict(dimportance)
            if param["objective"] in ("reg:squarederror", "reg:linear"):
                loss_sum += mean_squared_error(dimportance.get_label(), predictions)
            else:
                loss_sum += 1 - roc_auc_score(dimportance.get_label(), predictions)

        importances[idx] = baseline - loss_sum / perm_round

    return importances, time() - time_begin


def validate_total_gain(
    candidate: np.ndarray, dtrain: xgb.DMatrix, param: Dict, num_boost_round: int
) -> float:
    dtrain.set_base_margin([])

    bst_all_in_one = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)
    assert bst_all_in_one.feature_names is not None
    total_gain_dict = bst_all_in_one.get_score(importance_type="total_gain")
    total_gain = np.array(
        [total_gain_dict.get(f, 0) for f in bst_all_in_one.feature_names]
    ).astype('float64')

    total_gain /= np.sum(total_gain)
    candidate /= np.sum(candidate)

    return np.max(np.abs(total_gain - candidate))
