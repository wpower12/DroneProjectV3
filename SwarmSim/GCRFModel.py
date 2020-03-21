import numpy as np
from . import constants as C
import math

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin


class MyMultiOutputRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, num_outs, rnd_state):
        self.num_outputs = num_outs
        self.output_regressors = []
        for k in range(self.num_outputs):
            # self.output_regressors.append( MLPRegressor(random_state=rnd_state, hidden_layer_sizes=(50,50), alpha=0.04, solver='lbfgs', shuffle=False) )
            self.output_regressors.append(
                MLPRegressor(random_state=rnd_state, hidden_layer_sizes=(30,), alpha=0.015, solver='lbfgs',
                             shuffle=False))

    # self.output_regressors.append( MLPRegressor(random_state=rnd_state, hidden_layer_sizes=(30,), alpha=0.005, solver='lbfgs', shuffle=False) )

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        for k in range(self.num_outputs):
            self.output_regressors[k].fit(X, y[:, k])
        return self

    def partial_fit(self, X, y):
        for k in range(self.num_outputs):
            self.output_regressors[k].partial_fit(X, y[:, k])
        return self

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.num_outputs), dtype=float)
        for k in range(self.num_outputs):
            preds[:, k] = self.output_regressors[k].predict(X)

        return preds


class GCRFModel():
    def __init__(self, M=None, rnd_state=None):

        ### Weak Learners ###
        self.rnd_state = rnd_state

        self.M = M
        self.all_weak_learners = []
        self.thetas = []

    def train_single_GCRF(self, S_unfolded, X, Y, use_structure):

        # print('==== train() ====')
        # print(S.shape)
        # print(X.shape)
        # print(Y.shape)

        self.D = len(X)
        self.d_out = Y[0].shape[1]
        self.T = X[0].shape[0]
        K = 1

        weak_learners = []
        for ind in range(0, self.D):
            weak_learners.append(MyMultiOutputRegressor(self.d_out, self.rnd_state))
            weak_learners[-1].fit(X[ind], Y[ind])

        self.all_weak_learners.append(weak_learners)

        # =========================================================================

        if use_structure:

            R_unfolded = np.zeros((self.T * self.D, self.d_out), dtype=float)
            for ind in range(0, self.D):
                preds = weak_learners[ind].predict(X[ind])
                for t in range(0, self.T):
                    R_unfolded[t * self.D + ind, :] = preds[t, :]

            Y_unfolded = np.zeros((self.T * self.D, self.d_out), dtype=float)
            for ind in range(0, self.D):
                for t in range(0, self.T):
                    Y_unfolded[t * self.D + ind, :] = Y[ind][t, :]

            # S_unfolded = flatten_S(S)

            # Temporal GCRF training
            # print('Temporal GCRF training ...')

            S_unfolded = (S_unfolded / sum(sum(S_unfolded))) * S_unfolded.shape[0]
            L_unfolded = np.diag(sum(S_unfolded)) - S_unfolded

            # Initialize params
            alpha = np.ones(K)
            beta = 0
            params = np.append(alpha, beta)

            cons = ({'type': 'ineq', 'fun': lambda params: sum(params[0:K])},
                    {'type': 'ineq', 'fun': lambda params: params[K]})

            res = minimize(GCRF_objective, params, args=(L_unfolded, R_unfolded, Y_unfolded),
                           # jac=GCRF_objective_deriv,
                           constraints=cons, method='SLSQP', options={'maxiter': 20, 'disp': False})

            self.thetas.append(res.x)


    def train(self, S_unfolded, X, Y, use_structure):

        for m in range(self.M):
            self.train_single_GCRF(S_unfolded, X, Y, use_structure)




    def predict_single_GCRF(self, X, S, use_structure, gcrf_ind):

        # pred = self.weak_learners[d_ind].predict( X[d_ind].reshape(1,-1) )
        # dim_out = pred.shape[1]
        # pred = pred.reshape(dim_out)
        # return pred

        R = np.zeros((X.shape[0], self.d_out), dtype=float)

        for ind in range(0, self.D):
            pred = self.all_weak_learners[gcrf_ind][ind].predict(X[ind, :].reshape(1, -1))
            R[ind, :] = pred

        if not use_structure:
            return R, np.zeros(R.shape)

        # =========================================================================

        # mu <- GCRF_PREDICT(theta, S, R)
        N = S.shape[0]
        K = 1

        S = (S / sum(sum(S))) * N
        L = np.diag(sum(S)) - S

        alpha = self.thetas[gcrf_ind][0:K]
        gamma = sum(alpha)
        beta = self.thetas[gcrf_ind][K]

        Q = beta * L + gamma * np.eye(N)

        # mu = np.linalg.solve(Q,R*alpha)
        Q_inv = np.linalg.pinv(Q)
        mu = np.dot(Q_inv, R * alpha)

        return mu


    def predict(self, X, S, use_structure):

        preds = np.zeros((X.shape[0], self.d_out, self.M), dtype=float)
        for m in range(self.M):
            preds[:,:,m] = self.predict_single_GCRF(X, S, use_structure, m)
        pred_means = np.mean(preds, axis=2)
        pred_stds = np.std(preds, axis=2)
        return pred_means, pred_stds



def GCRF_objective(params, L, R, Y):
    (TN, d_out) = R.shape
    K = 1
    alpha = params[0:K]
    beta = params[K]
    epsilon = 0.0  # 1e-8

    gamma = sum(alpha)
    Q = beta * L + gamma * np.eye(TN)
    # Q_inv = np.linalg.inv(Q)
    Q_inv = np.linalg.pinv(Q)  # Optional

    neg_ll = 0
    for j in range(0, d_out):
        b = R[:, j] * alpha
        # mu = np.linalg.solve(Q,b)
        mu = np.dot(Q_inv, b)
        e = Y[:, j] - mu
        neg_ll = neg_ll - np.dot(np.dot(e.T, Q), e) - 0.5 * np.log(np.linalg.det(Q_inv) + epsilon)
    neg_ll = -neg_ll

    return neg_ll


def flatten_S(S_tensor):
    [N, _, T] = S_tensor.shape
    S_mat = np.zeros((T * N, T * N), dtype=float)
    for t in range(0, T):
        S_mat[t * N:t * N + N, t * N:t * N + N] = S_tensor[:, :, t]
    return S_mat

