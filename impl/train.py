"""
Module for model training and evaluation.
"""
import slim.slim_python as slim
from gosdt.model.gosdt import GOSDT
from gosdt.model.threshold_guess import compute_thresholds, cut

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from interpret.glassbox import ExplainableBoostingClassifier
from catboost import CatBoostClassifier

from preparation import *
from config import *

warnings.filterwarnings("ignore", category=UserWarning)


def train_model(model, Xtrain, Xval, ytrain, yval, param_g):
    """
    Search through a parameter grid to find best performing hyperparameters
    :param model: model name
    :param Xtrain: features matrix used to train the model
    :param Xval: features matrix used to select the best performing model
    :param ytrain: target variable used to train the model
    :param yval: target variable for model evaluation
    :param param_g: parameter grid to search
    :return: best performing hyperparameters along with precision, recall and f1 score
    """
    f1_v = 0
    best = dict()
    rep = None
    # find the hyperparameters with the best f1 score on the validation set
    for params in param_g:
        clf = model(**params)
        clf.fit(Xtrain, ytrain)
        yhat = clf.predict(Xval)
        f1 = f1_score(yval, yhat)
        if f1 > f1_v:
            f1_v = f1
            best = params
            rep = classification_report(yval, yhat)
    return best, rep


def train_final(model, params, Xtrain, Xtest, ytrain, ytest):
    """
    Train final model to provide performance metrics on unseen data
    :param model: model name
    :param params: hyperparameters to train the model with
    :param Xtrain: features matrix used to train the model
    :param Xtest: features matrix used to estimate model performance on unseen data
    :param ytrain: target variable used to train the model
    :param ytest: target variable for performance on unseen data
    :return: final model along with its confusion matrix, precision, recall and f1 score
    """
    clf = model(**params)
    clf.fit(Xtrain, ytrain)
    yhat = clf.predict(Xtest)
    tn, fp, fn, tp = confusion_matrix(ytest, yhat).ravel()
    conf_mat = np.array([[tp, fp], [fn, tn]])
    rep = classification_report(ytest, yhat)
    return clf, conf_mat, rep


def train_gam():
    """
    Find best performing gam for each dataset.
    :return: models
    """
    def find_gam(Xtrain, Xval, ytrain, yval, params_g=gam_params):
        """
        Evaluate multiple gams and select the best performing one.
        :param Xtrain: features matrix used to train the model
        :param Xval: features matrix used to select the best performing model
        :param ytrain: target variable used to train the model
        :param yval: target variable for model evaluation
        :param params_g: hyperparameter values to explore
        :return: best performing hyperparameters along with precision, recall and f1 score
        """
        pg = ParameterGrid(params_g)
        return train_model(ExplainableBoostingClassifier, Xtrain, Xval, ytrain, yval, pg)

    X_train, X_val, X_test, y_train, y_val, y_test = load_credit()
    cred_p, report = find_gam(X_train, X_val, y_train, y_val)
    credit = {"params": cred_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_heart()
    heart_p, report = find_gam(X_train, X_val, y_train, y_val)
    heart = {"params": heart_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_hypothyroid()
    hypo_p, report = find_gam(X_train, X_val, y_train, y_val)
    hypo = {"params": hypo_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_euthyroid()
    euth_p, report = find_gam(X_train, X_val, y_train, y_val)
    euth = {"params": euth_p, "report": report}
    return credit, heart, hypo, euth


def train_final_gam(params, Xtrain, Xtest, ytrain, ytest):
    """
    Find final gam to provide performance metrics on unseen data.
    :param params: hyperparameters to train the model with
    :param Xtrain: features matrix used to train the model
    :param Xtest: features matrix used to estimate model performance on unseen data
    :param ytrain: target variable used to train the model
    :param ytest: target variable for performance on unseen data
    :return: confusion matrix, precision, recall and f1 score
    """
    return train_final(ExplainableBoostingClassifier, params, Xtrain, Xtest, ytrain, ytest)


def binarize_gosdt(Xtrain, Xval, ytrain, n_est=40, max_depth=1):
    """
    Transform datasets using an ensemble of trees to speed up training
    :param Xtrain: features matrix used to train the model
    :param Xval: features matrix used to select the best performing model
    :param ytrain: target variable used to train the model
    :param n_est: number of trees in ensemble
    :param max_depth: depth of the used trees
    :return: transformed datasets
    """
    X_train_g, thresholds, header, _ = compute_thresholds(Xtrain, ytrain, n_est, max_depth)
    X_val_g = cut(Xval, thresholds)
    X_val_g = X_val_g[header]
    return X_train_g, X_val_g


def train_gosdt():
    """
    Find best performing decision tree for each dataset.
    :return: models
    """
    def find_gosdt(Xtrain, Xval, ytrain, yval, params_g=gosdt_params):
        """
        Evaluate multiple decision trees and select the best performing one.
        :param Xtrain: features matrix used to train the model
        :param Xval: features matrix used to select the best performing model
        :param ytrain: target variable used to train the model
        :param yval: target variable for model evaluation
        :param params_g: hyperparameter values to explore
        :return: best performing hyperparameters along with precision, recall and f1 score.
        """
        pg = ParameterGrid(params_g)
        f1_v = 0
        best = dict()
        rep = None
        # prepare datasets
        X_train_g, X_val_g = binarize_gosdt(Xtrain.copy(), Xval.copy(), ytrain)
        # find the hyperparameters with the best f1 score on the validation set
        for params in pg:
            clf = GOSDT(params)
            clf.fit(X_train_g, ytrain.values.flatten())
            yhat = clf.predict(X_val_g)
            f1 = f1_score(yval, yhat)
            if f1 > f1_v:
                f1_v = f1
                best = params
                rep = classification_report(yval, yhat)
        return best, rep

    X_train, X_val, X_test, y_train, y_val, y_test = load_credit()
    cred_p, report = find_gosdt(X_train, X_val, y_train, y_val)
    credit = {"params": cred_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_heart_nonan()
    heart_p, report = find_gosdt(X_train, X_val, y_train, y_val)
    heart = {"params": heart_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_hypo_nonan()
    hypo_p, report = find_gosdt(X_train, X_val, y_train, y_val)
    hypo = {"params": hypo_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_euth_nonan()
    euth_p, report = find_gosdt(X_train, X_val, y_train, y_val)
    euth = {"params": euth_p, "report": report}
    return credit, heart, hypo, euth


def train_final_gosdt(params, Xtrain, Xtest, ytrain, ytest):
    """
    Find final sparse decision tree to provide performance metrics on unseen data.
    :param params: hyperparameters to train the model with
    :param Xtrain: features matrix used to train the model
    :param Xtest: features matrix used to estimate model performance on unseen data
    :param ytrain: target variable used to train the model
    :param ytest: target variable for performance on unseen data
    :return: confusion matrix, precision, recall and f1 score
    """
    X_train_g, X_test_g = binarize_gosdt(Xtrain.copy(), Xtest.copy(), ytrain)
    clf = GOSDT(params)
    clf.fit(X_train_g, ytrain.values.flatten())
    yhat = clf.predict(X_test_g)
    tn, fp, fn, tp = confusion_matrix(ytest, yhat).ravel()
    conf_mat = np.array([[tp, fp], [fn, tn]])
    rep = classification_report(ytest, yhat)
    return clf, conf_mat, rep


def get_coefs(Xtrain, ytrain, names):
    """
    Prepare coefficient constraints used to train the model.
    :param Xtrain: features matrix used to train the model
    :param ytrain: target variable used to train the model
    :param names: feature names
    :return: prepared coefficients
    """
    # src: https://github.com/ustunb/slim-python/blob/master/demos/ex_1_basic_functionality.py
    coefs = slim.SLIMCoefficientConstraints(variable_names=names, ub=5, lb=-5)
    scores_at_ub = (ytrain * Xtrain) * coefs.ub
    scores_at_lb = (ytrain * Xtrain) * coefs.lb
    non_intercept_ind = np.array([n != '(Intercept)' for n in names])
    scores_at_ub = scores_at_ub[:, non_intercept_ind]
    scores_at_lb = scores_at_lb[:, non_intercept_ind]
    max_scores = np.fmax(scores_at_ub, scores_at_lb)
    min_scores = np.fmin(scores_at_ub, scores_at_lb)
    max_scores = np.sum(max_scores, 1)
    min_scores = np.sum(min_scores, 1)
    intercept_ub = -min(min_scores) + 1
    intercept_lb = -max(max_scores) + 1
    coefs.set_field('ub', '(Intercept)', intercept_ub)
    coefs.set_field('lb', '(Intercept)', intercept_lb)
    return coefs


def prepare_params(slim_IP, time=60.0, threads=8, verbose=0):
    """
    Prepare solver parameters
    :param slim_IP: SLIM internal parameters
    :param time: time limit
    :param threads: number of threads you want to use
    :param verbose: print out each solver step
    :return: prepared solver parameters
    """
    slim_IP.parameters.timelimit.set(time)
    slim_IP.parameters.randomseed.set(RAND_VAL)
    slim_IP.parameters.threads.set(threads)
    slim_IP.parameters.parallel.set(1)
    slim_IP.parameters.output.clonelog.set(0)
    slim_IP.parameters.mip.tolerances.mipgap.set(np.finfo(np.cfloat).eps)
    slim_IP.parameters.mip.tolerances.absmipgap.set(np.finfo(np.cfloat).eps)
    slim_IP.parameters.mip.tolerances.integrality.set(np.finfo(np.cfloat).eps)
    slim_IP.parameters.mip.display.set(verbose)
    slim_IP.parameters.emphasis.mip.set(1)
    return slim_IP


def train_slim():
    """
    Find best performing supersparse linear integer model for each dataset.
    :return: models
    """
    def find_slim(Xtrain, Xval, ytrain, yval, params_g=slim_params):
        """
        Evaluate multiple SLIMs and select the best performing one.
        :param Xtrain: features matrix used to train the model
        :param Xval: features matrix used to select the best performing model
        :param ytrain: target variable used to train the model
        :param yval: target variable for model evaluation
        :param params_g: hyperparameter values to explore
        :return: best performing hyperparameters, confusion matrix, precision, recall and f1 score.
        """
        param_g = ParameterGrid(params_g)
        # weights to combat imbalanced dataset
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(ytrain), y=ytrain.values.reshape(-1))
        w_sum = weights[0] + weights[1]
        neg_weight = 2 * weights[0] / w_sum
        pos_weight = 2 * weights[1] / w_sum
        # minor dataset tweaks to be able to use and interpret the resulting SLIM
        X_train_g, X_val_g = binarize_gosdt(Xtrain.copy(), Xval.copy(), ytrain)
        X_train_g.insert(0, '(Intercept)', 1)
        X_val_g.insert(0, '(Intercept)', 1)
        X_names = list(X_train_g.columns.values)
        y_name = ytrain.name
        ytrain[ytrain == 0] = -1
        Xtrain = X_train_g.to_numpy()
        Xval = X_val_g.to_numpy()
        ytrain = ytrain.to_numpy().reshape(-1, 1)
        yval = yval.to_numpy().reshape(-1, 1)
        # slim coefficient setup
        slim.check_data(X=Xtrain, Y=ytrain, X_names=X_names)
        coefs = get_coefs(Xtrain, ytrain, X_names)
        f1_v = 0
        best = dict()
        rep = None
        # find the hyperparameters with the best f1 score on the validation set
        for params in param_g:
            slim_input = {
                'X': Xtrain,
                'X_names': X_names,
                'Y': ytrain,
                'Y_name': y_name,
                'C_0': params["C_0"],
                'w_pos': pos_weight,
                'w_neg': neg_weight,
                'L0_min': 0,
                'L0_max': float('inf'),
                'err_min': 0,
                'err_max': min(pos_weight, neg_weight),
                'pos_err_min': 0,
                'pos_err_max': pos_weight,
                'neg_err_min': 0,
                'neg_err_max': neg_weight,
                'coef_constraints': coefs
            }
            slim_IP, slim_info = slim.create_slim_IP(slim_input)
            slim_IP = prepare_params(slim_IP)
            # solve SLIM IP
            slim_IP.solve()
            slim_results = slim.get_slim_summary(slim_IP, slim_info, Xval, yval)
            # get predictions
            yhat = Xval.dot(slim_results["rho"]) > 0
            yhat = np.array(yhat, dtype="d")
            f1 = f1_score(yval, yhat)
            if f1 > f1_v:
                f1_v = f1
                best = params
                rep = classification_report(yval, yhat)
        return best, rep, weights
    X_train, X_val, X_test, y_train, y_val, y_test = load_credit()
    cred_p, report, cred_w = find_slim(X_train, X_val,  y_train, y_val)
    credit = {"params": cred_p, "report": report, "weights": cred_w}
    X_train, X_val, X_test, y_train, y_val, y_test = load_heart_nonan()
    heart_p, report, heart_w = find_slim(X_train, X_val,  y_train, y_val)
    heart = {"params": heart_p, "report": report, "weights": heart_w}
    X_train, X_val, X_test, y_train, y_val, y_test = load_hypo_nonan()
    hypo_p, report, hypo_w = find_slim(X_train, X_val, y_train, y_val)
    hypo = {"params": hypo_p, "report": report, "weights": hypo_w}
    X_train, X_val, X_test, y_train, y_val, y_test = load_euth_nonan()
    euth_p, report, euth_w = find_slim(X_train, X_val, y_train, y_val)
    euth = {"params": euth_p, "report": report, "weights": euth_w}
    return credit, heart, hypo, euth


def train_final_slim(params, weights, Xtrain, Xtest, ytrain, ytest):
    """
    Find final sparse decision tree to provide performance metrics on unseen data.
    :param params: hyperparameters to train the model with
    :param weights: weight of each observation depending on its class
    :param Xtrain: features matrix used to train the model
    :param Xtest: features matrix used to estimate model performance on unseen data
    :param ytrain: target variable used to train the model
    :param ytest: target variable for performance on unseen data
    :return: final model along with its confusion matrix, precision, recall and f1 score
    """
    w_sum = weights[0] + weights[1]
    neg_weight = 2 * weights[0] / w_sum
    pos_weight = 2 * weights[1] / w_sum
    # minor dataset tweaks to be able to use and interpret the resulting SLIM
    X_train_g, X_test_g = binarize_gosdt(Xtrain.copy(), Xtest.copy(), ytrain)
    X_train_g.insert(0, '(Intercept)', 1)
    X_test_g.insert(0, '(Intercept)', 1)
    X_names = list(X_train_g.columns.values)
    y_name = ytrain.name
    ytrain[ytrain == 0] = -1
    Xtrain = X_train_g.to_numpy()
    Xtest = X_test_g.to_numpy()
    ytrain = ytrain.to_numpy().reshape(-1, 1)
    ytest = ytest.to_numpy().reshape(-1, 1)
    slim.check_data(X=Xtrain, Y=ytrain, X_names=X_names)
    coefs = get_coefs(Xtrain, ytrain, X_names)
    slim_input = {
            'X': Xtrain,
            'X_names': X_names,
            'Y': ytrain,
            'Y_name': y_name,
            'C_0': params["C_0"],
            'w_pos': pos_weight,
            'w_neg': neg_weight,
            'L0_min': 0,
            'L0_max': float('inf'),
            'err_min': 0,
            'err_max': min(pos_weight, neg_weight),
            'pos_err_min': 0,
            'pos_err_max': pos_weight,
            'neg_err_min': 0,
            'neg_err_max': neg_weight,
            'coef_constraints': coefs
    }
    slim_IP, slim_info = slim.create_slim_IP(slim_input)
    slim_IP = prepare_params(slim_IP)
    # solve SLIM IP
    slim_IP.solve()
    slim_results = slim.get_slim_summary(slim_IP, slim_info, Xtest, ytest)
    # get predictions
    yhat = Xtest.dot(slim_results["rho"]) > 0
    yhat = np.array(yhat, dtype="d")
    tn, fp, fn, tp = confusion_matrix(ytest, yhat).ravel()
    conf_mat = np.array([[tp, fp], [fn, tn]])
    rep = classification_report(ytest, yhat)
    return slim_results, conf_mat, rep


def train_cat():
    """
    Find best performing ensemble of boosted trees for each dataset.
    :return: models
    """
    def find_cat(Xtrain, Xval, ytrain, yval, params_g=cat_params):
        """
        Evaluate many boosted tree ensembles and select the best performing one.
        :param Xtrain: features matrix used to train the model
        :param Xval: features matrix used to select the best performing model
        :param ytrain: target variable used to train the model
        :param yval: target variable for model evaluation
        :param params_g: hyperparameter values to explore
        :return: best performing hyperparameters along with precision, recall and f1 score
        """
        # let catboost try to tune the hyperparameters itself (other than the random value)
        params_g = [params_g, {"verbose": [False], "random_state": [RAND_VAL]}]
        pg = ParameterGrid(params_g)
        return train_model(CatBoostClassifier, Xtrain, Xval, ytrain, yval, pg)

    X_train, X_val, X_test, y_train, y_val, y_test = load_credit()
    cred_p, report = find_cat(X_train, X_val, y_train, y_val)
    credit = {"params": cred_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_heart()
    heart_p, report = find_cat(X_train, X_val, y_train, y_val)
    heart = {"params": heart_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_hypothyroid()
    hypo_p, report = find_cat(X_train, X_val, y_train, y_val)
    hypo = {"params": hypo_p, "report": report}
    X_train, X_val, X_test, y_train, y_val, y_test = load_euthyroid()
    euth_p, report = find_cat(X_train, X_val,  y_train, y_val)
    euth = {"params": euth_p, "report": report}
    return credit, heart, hypo, euth


def train_final_cat(params, Xtrain, Xtest, ytrain, ytest):
    """
    Find final ensemble of boosted trees to provide performance metrics on unseen data.
    :param params: hyperparameters to train the model with
    :param Xtrain: features matrix used to train the model
    :param Xtest: features matrix used to estimate model performance on unseen data
    :param ytrain: target variable used to train the model
    :param ytest: target variable for performance on unseen data
    :return: final model along with its confusion matrix, precision, recall and f1 score
    """
    return train_final(CatBoostClassifier, params, Xtrain, Xtest, ytrain, ytest)