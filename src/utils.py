from time import time
from math import isclose
import numpy as np
from scipy.special import expit
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb
import warnings


transformation = {
    "reg:squarederror": lambda margins: margins,
    "reg:linear": lambda margins: margins,
    "binary:logitraw": expit,
    "reg:logistic": expit,
    "binary:logistic": expit,
}


def feature_importance(
    Y_train,
    X_train,
    Y_valid,
    X_valid,
    param,
    num_boost_round=1,
    use_saabas=True,
    do_sanity_check=True,
    verbosity=0,
):
    # sanity check
    for name in ["max_depth", "eta", "base_score", "objective", "reg_lambda"]:
        assert name in param, "{} should be in param.".format(name)

    time_begin = time()

    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)

    # train a boosting forest with DTRAIN, use it to make prediction for DVALID,
    # and decompose the result into feature contributions
    contributions_by_tree = _compute_contribution(
        dtrain, dvalid, param, num_boost_round, use_saabas, do_sanity_check
    )

    # compute gradient
    gradient_by_tree = _compute_gradient(
        dvalid,
        contributions_by_tree,
        param["objective"],
        param["base_score"],
        num_boost_round,
    )
    if verbosity >= 3:
        print(gradient_by_tree)

    gradient_by_tree_reshaped = np.reshape(
        gradient_by_tree, (*gradient_by_tree.shape, 1)
    )
    MDI = np.sum(contributions_by_tree * gradient_by_tree_reshaped, (0, 1))
    MDI = MDI[:-1] / param["eta"]

    coef = np.empty((num_boost_round, dvalid.num_col() + 1))    # last column is for bias/intercept
    for t in range(num_boost_round):
        clf = linear_model.LinearRegression()
        contributions = contributions_by_tree[t, :, :-1]
        gradients = gradient_by_tree[t, :]
        clf.fit(contributions, gradients)
        coef[t, :-1] = clf.coef_
        coef[t, -1] = clf.intercept_

    if do_sanity_check and Y_train is Y_valid and X_train is X_valid and use_saabas:
        bst_all_in_one = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)
        total_gain_dict = bst_all_in_one.get_score(importance_type="total_gain")
        total_gain_dict = {k.strip(): v for k, v in total_gain_dict.items()}
        for i, mdi in enumerate(MDI):
            arg1 = mdi
            arg2 = total_gain_dict.get(str(i), 0)
            if not isclose(arg1, arg2, rel_tol=1e-4, abs_tol=1e-4):
                warnings.warn(
                    f"  MDI for feature #{i} should be the same: arg1 = {arg1}, arg2 = {arg2}, when param = {param}."
                )

    return MDI, coef, time() - time_begin


def _compute_contribution(
    dtrain, dvalid, param, num_boost_round, use_saabas, do_sanity_check
):
    # reset the margin of dtrain & dvalid
    dtrain.set_base_margin([])
    dvalid.set_base_margin([])

    # store the feature contribution of each tree
    contributions_by_tree = np.zeros(
        (num_boost_round, dvalid.num_row(), dvalid.num_col() + 1), dtype=np.float32
    )

    for t in range(num_boost_round):
        # train a new tree
        bst = xgb.train(param, dtrain, 1, verbose_eval=False)

        # compute the contribution_by_tree
        if use_saabas:
            pvalid_tree = bst.predict(
                dvalid,
                pred_contribs=True,
                approx_contribs=True,
                reg_lambda=param["reg_lambda"],
            )
        else:
            pvalid_tree = bst.predict(dvalid, pred_contribs=True)

        # sanity check
        if do_sanity_check:
            arg1 = np.sum(pvalid_tree, axis=1)
            arg2 = bst.predict(dvalid, output_margin=True)
            if not np.allclose(arg1, arg2, rtol=1e-5, atol=1e-5):
                warnings.warn(
                    f"  Contributions should sum to prediction at round {t+1}/{num_boost_round}: error = {max(abs(arg1 - arg2))}, when param = {param}."
                )

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
        dvalid.set_base_margin(np.sum(pvalid_tree, axis=1))

    if do_sanity_check:
        # checks if $f_{m,k}$ and the bias add up to the prediction of trees
        arg1 = np.sum(contributions_by_tree, axis=(0, 2))
        arg2 = np.sum(pvalid_tree, axis=1)
        if not np.allclose(arg1, arg2, rtol=1e-5, atol=1e-5):
            warnings.warn(
                f"  Contributions should sum to prediction: error = {max(abs(arg1 - arg2))}, when param = {param} and num_boost_round = {num_boost_round}."
            )

        # checks if training individual trees from staged predictions is the same as training the entire tree ensemble as a whole,
        # a.k.a. checks if boosting from prediction is done right.
        dtrain.set_base_margin([])  # reset base margin
        dvalid.set_base_margin([])  # reset base margin
        bst_all_in_one = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)
        pvalid_all_in_one = bst_all_in_one.predict(dvalid, output_margin=True)
        arg1 = np.sum(contributions_by_tree, axis=(0, 2))
        arg2 = pvalid_all_in_one
        if not np.allclose(arg1, arg2, rtol=1e-5, atol=1e-5):
            warnings.warn(
                f"  Predictions should be close: error = {max(abs(arg1 - arg2))}, when param = {param} and num_boost_round = {num_boost_round}."
            )

    return contributions_by_tree


def _compute_gradient(
    dvalid, contributions_by_tree, objective, base_score, num_boost_round
):
    # compute gradient for each tree
    gradient_by_tree = np.zeros((num_boost_round, dvalid.num_row()), dtype=np.float32)
    gradient_by_tree[0, :] = dvalid.get_label() - base_score

    # TODO: is this correct for classification?
    trans = transformation[objective]
    for t in range(1, num_boost_round):
        gradient_by_tree[t, :] = dvalid.get_label() - trans(
            np.sum(contributions_by_tree[:t, :, :], axis=(0, 2))
        )

    return gradient_by_tree


def evaluate_total_gain(
    total_gain, Y_full, X_full, param, num_boost_round, verbosity=0
):
    dtrain = xgb.DMatrix(X_full, Y_full, silent=True)  # reset base margin
    bst_all_in_one = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)
    total_gain_dict = bst_all_in_one.get_score(importance_type="total_gain")
    arg1 = total_gain
    arg2 = np.array([total_gain_dict.get(f, 0) for f in bst_all_in_one.feature_names])
    if not np.allclose(arg1, arg2, rtol=1e-4, atol=1e-4):
        warnings.warn(
            f"  Total gains should be close: error = {max(abs(arg1 - arg2))}, when param = {param} and num_boost_round = {num_boost_round}."
        )

    return max(abs(arg1 - arg2))


def permutation_importance(
    Y_train, X_train, Y_valid, X_valid, param, num_boost_round=1, perm_round=5
):
    # sanity check
    for name in ["max_depth", "eta", "base_score", "objective", "reg_lambda"]:
        assert name in param, "{} should be in param.".format(name)

    time_begin = time()

    # we don't want to mess with the original object
    X_valid = X_valid.copy()

    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)

    # train a boosting forest with DTRAIN, which will be used to make prediction for DVALID
    bst = xgb.train(param, dtrain, num_boost_round, verbose_eval=False)

    # calculate baseline prediction accuracy
    predictions = bst.predict(dvalid)
    if param["objective"] in ("reg:squarederror", "reg:linear"):
        baseline = mean_squared_error(dvalid.get_label(), predictions)
    else:
        baseline = 1 - roc_auc_score(dvalid.get_label(), predictions)

    importances = np.zeros((len(X_valid.columns),), dtype=np.float32)
    for idx, feature in enumerate(X_valid.columns):
        # permute row and re-calculate prediction accuracy
        loss_sum = 0
        for _ in range(perm_round):
            save = X_valid[feature].copy()
            X_valid[feature] = np.random.permutation(X_valid[feature])
            dvalid = xgb.DMatrix(X_valid, Y_valid, silent=True)
            X_valid[feature] = save

            predictions = bst.predict(dvalid)
            if param["objective"] in ("reg:squarederror", "reg:linear"):
                loss_sum += mean_squared_error(dvalid.get_label(), predictions)
            else:
                loss_sum += 1 - roc_auc_score(dvalid.get_label(), predictions)

        importances[idx] = baseline - loss_sum / perm_round

    return importances, time() - time_begin


def generalization_error(
    Y_train, X_train, Y_test, X_test, param, num_boost_round, coef=None, use_saabas=True, do_sanity_check=True
):
    # sanity check
    for name in ["max_depth", "eta", "base_score", "objective", "reg_lambda"]:
        assert name in param, "{} should be in param.".format(name)

    # build DMatrix from data frames
    dtrain = xgb.DMatrix(X_train, Y_train, silent=True)
    dtest = xgb.DMatrix(X_test, Y_test, silent=True)

    contributions_by_tree = _compute_contribution(
        dtrain, dtest, param, num_boost_round, use_saabas, do_sanity_check
    )
    if coef is not None:
        contributions_by_tree[:, :, :-1] *= coef[:, np.newaxis, :-1]
        contributions_by_tree[:, :, -1] = coef[:, np.newaxis, -1]

    margins = np.sum(contributions_by_tree, axis=(0, 2))
    trans = transformation[param["objective"]]
    predictions = trans(margins)

    if param["objective"] in ("reg:squarederror", "reg:linear"):
        return mean_squared_error(dtest.get_label(), predictions)
    else:
        return 1 - roc_auc_score(dtest.get_label(), predictions)
