import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from GP_kernel import KernelFunctions


class GaussianProcess:
    def __init__(self, kernel_type='SE'):
        self.kernel_obj = KernelFunctions()
        self.kernel_type = kernel_type

    def kernel(self, X1, X2, *args, **kwargs):
        if self.kernel_type == 'SE':
            return self.kernel_obj.SE_kernel(X1, X2, *args, **kwargs)
        elif self.kernel_type == 'ARD':
            return self.kernel_obj.ARD_kernel(X1, X2, *args, **kwargs)

    def posterior(
        self, X_s, X_train, Y_train, length_scale=1.0, sigma_f=1.0,
        sigma_y=1e-8
    ):
        K = self.kernel(X_train, X_train, length_scale, sigma_f)
        K += sigma_y**2 * np.eye(len(X_train))
        K_s = self.kernel(X_train, X_s, length_scale, sigma_f)
        K_ss = self.kernel(X_s, X_s, length_scale, sigma_f)
        K_ss + 1e-8 * np.eye(len(X_s))
        K_inv = inv(K)

        mu_s = K_s.T.dot(K_inv).dot(Y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, cov_s

    def distributed_posterior(
        self, X_s, X_train, Y_train, length_scale=1.0, sigma_f=1.0,
        sigma_y=1e-8
    ):
        mu_s = [None] * len(X_train)
        cov_s = [None] * len(X_train)

        for i in range(len(X_train)):
            K = self.kernel(X_train[i], X_train[i], length_scale, sigma_f)
            K += sigma_y**2 * np.eye(len(X_train[i]))
            K_s = self.kernel(X_train[i], X_s, length_scale, sigma_f)
            K_ss = self.kernel(X_s, X_s, length_scale, sigma_f)
            K_ss += 1e-8 * np.eye(len(X_s))
            K_inv = inv(K)
            Y_train1 = np.array(Y_train[i]).ravel()

            mu_s[i] = K_s.T.dot(K_inv).dot(Y_train1).reshape(-1, 1)
            cov_s[i] = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, cov_s

    def plot_gp(self, mu, cov, Xt, X_train=None, Y_train=None, samples=[]):
        Xt = Xt.ravel()
        mu = mu.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(cov))

        plt.fill_between(Xt, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(Xt, mu, label='Mean')
        for i, sample in enumerate(samples):
            plt.plot(Xt, sample, lw=1, ls='--', label=f'Sample {i+1}')
        if X_train is not None:
            plt.plot(X_train, Y_train, 'rx')
        plt.legend()

    def nll_fn(self, X_train, Y_train, flag):
        if flag == 'DGP':
            def nll_distributed(theta):
                object_fun = 0
                for i in range(len(X_train)):
                    Y_train1 = np.array(Y_train[i]).ravel()
                    K = self.kernel(
                        X_train[i], X_train[i], length_scale=theta[0],
                        sigma_f=theta[1]
                    ) + theta[2]**2 * np.eye(len(X_train[i]))

                    L = cholesky(K)

                    S1 = solve_triangular(L, Y_train1, lower=True)
                    S2 = solve_triangular(L.T, S1, lower=False)

                    fun = np.sum(np.log(np.diagonal(L)))
                    funa = 0.5 * Y_train1.dot(S2)
                    func = 0.5 * len(X_train[i]) * np.log(2 * np.pi)
                    object_fun += fun + funa + func
                return object_fun

            return nll_distributed
        elif flag == 'GP':
            def nll_stable(theta):
                Y_train1 = np.array(Y_train).ravel()
                K = self.kernel(
                    X_train, X_train, length_scale=theta[0],
                    sigma_f=theta[1]
                )
                K += theta[2]**2 * np.eye(len(X_train))
                L = cholesky(K)

                S1 = solve_triangular(L, Y_train1, lower=True)
                S2 = solve_triangular(L.T, S1, lower=False)
                fun = np.sum(np.log(np.diagonal(L)))
                funa = 0.5 * Y_train1.dot(S2)
                funb = 0.5 * len(X_train) * np.log(2 * np.pi)
                fun += funa + funb
                return fun

            return nll_stable

    def fit(self, X_train, Y_train, flag):
        return minimize(self.nll_fn(X_train, Y_train, flag), [1, 1, 0.1],
                        bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                        method='L-BFGS-B')

    def aggregation(self, mu_set, cov_set, X_t):
        s2 = np.zeros((len(X_t), 1))
        mu1 = s2

        for i in range(len(mu_set)):
            for j in range(len(mu_set[i])):
                s2 += 1 / cov_set[i][j, j]
        s2 = 1 / s2
        for i in range(len(mu_set)):
            for j in range(len(mu_set[i])):
                mu1 += s2 * (mu_set[i] / cov_set[i][j, j])

        return mu1, s2
