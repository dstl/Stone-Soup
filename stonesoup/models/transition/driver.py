import numpy as np
from scipy.special import gammainc, kv, hankel1
from scipy.special import gamma as gammafnc
from stonesoup.types.array import CovarianceMatrix, StateVector
from .base_driver import NormalSigmaMeanDriver, NormalVarianceMeanDriver
from ...base import Property
from typing import Optional


def incgammal(s: float, x: float) -> float: # Helper function
    return gammainc(s, x) * gammafnc(s)

class AlphaStableNSMDriver(NormalSigmaMeanDriver):
    alpha: float = Property(doc="Alpha parameter.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if np.isclose(self.alpha, 0.0) or np.isclose(self.alpha, 1.0) or np.isclose(self.alpha, 2.0):
            raise AttributeError("alpha must be 0 < alpha < 1 or 1 < alpha < 2.")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _first_moment(self, truncation: float) -> float:
        return self.alpha / (1. - self.alpha) * np.power(self.c, 1. - 1. / self.alpha)
    
    def _second_moment(self, truncation: float) -> float:
        return self.alpha / (2. - self.alpha) * np.power(self.c, 1. - 2. / self.alpha)

    def _residual_mean(self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None) -> StateVector:
        if 1 < self.alpha < 2:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
            return r_mean
        return super()._residual_mean(e_ft=e_ft, model_id=model_id, truncation=truncation)
    
    def _centering(self, e_ft: np.ndarray, truncation: float, model_id: Optional[int]= None) -> StateVector:
        if 1 < self.alpha < 2:
            term = e_ft * self._mu_W(model_id) # (m, 1)
            return - self._first_moment(truncation=truncation) * term # (m, 1)
        elif 0 < self.alpha < 1:
            m = e_ft.shape[0]
            return np.zeros((m, 1))
        else:
            raise AttributeError("alpha must be 0 < alpha < 2")

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        # accept all
        return np.ones_like(jsizes) # (n_jumps, n_samples)
    

class GammaNVMDriver(NormalVarianceMeanDriver):
    nu: float = Property(doc="Scale parameter")
    beta: float = Property(doc="Shape parameter")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return 1. / (self.beta * (np.exp(epochs / self.nu) - 1.))

    def _first_moment(self, truncation: float) -> float:
        return (self.nu / self.beta) * incgammal(1., self.beta * truncation)
    
    def _second_moment(self, truncation: float) -> float:
        return (self.nu / self.beta ** 2) * incgammal(2., self.beta * truncation)

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return (1. + self.beta * jsizes) * np.exp(-self.beta * jsizes) # (n_jumps, n_samples)

    def _residual_covar(self, e_ft: np.ndarray, truncation: float, model_id: int | None = None) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, m))
    
    def _residual_mean(self, e_ft: np.ndarray, truncation: float, model_id: int | None = None) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, 1))
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        truncation = self._hfunc(self.c)
        nu = 1 / self.nu  # shape
        # gamma = np.sqrt(2 * self.beta)
        sigma = np.sqrt(self.sigma_W2) # spread
        theta = self.mu_W # asymmetry
        # nu = 1 

        # theta = self.mu_W
        # print((self.mu_W ** 2 * self._second_moment(truncation) + self.sigma_W2 * self._first_moment(truncation)) ** 0.5)
        # sigma = (self.mu_W ** 2 * self._second_moment(truncation) + self.sigma_W2 * self._first_moment(truncation)) ** 0.5
        c = 0
        temp1 = 2.0 / ( sigma*(2.0*np.pi)**0.5*nu**(1/nu)*gammafnc(1/nu) )
        temp2 = ((2*sigma**2/nu+theta**2)**0.5)**(0.5-1/nu)
        temp3 = np.exp(theta*(x-c)/sigma**2) * abs(x-c)**(1/nu - 0.5)
        temp4 = kv(1/nu - 0.5, abs(x-c)*(2*sigma**2/nu+theta**2)**0.5/sigma**2)
        return temp1*temp2*temp3*temp4
        
    
        # beta = self.mu_W
        # gamma = np.sqrt((self.beta / self.sigma_W2) * 2)
        # alpha = np.sqrt(beta ** 2 + gamma ** 2)
        # # alpha = self.beta / self.sigma_W2
        # nu = self.nu
        # mu = 0

        # # constant
        # m = np.sqrt((2 * self.sigma_W2 * self.nu + self.mu_W ** 2) / (self.beta ** 2 * self.sigma_W2))

        # # Calculate the terms in the formula
        # term1 = (gamma ** (2 * nu)) * (alpha ** (1 - 2 * nu))
        # term2 = np.sqrt(2 * np.pi) * gammafnc(nu) * (2 ** (nu - 1))
        # term3 = (gamma ** 2 / 2) * m * np.abs(x - mu)
        # print(m)
        # term4 = nu - 0.5
        # term5 = (term3 ** term4) * kv(term4, term3)
        # term6 = term5 * np.exp(beta * (x - mu))
        
        # return term1 / term2 * term6


    

class TemperedStableNVMDriver(NormalVarianceMeanDriver):
    alpha: float = Property(doc="Alpha parameter")
    beta: float = Property(doc="Shape parameter")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)
    
    def _first_moment(self, truncation: float) -> float:
        return (self.alpha * self.beta ** (self.alpha - 1.)) * incgammal(1. - self.alpha, self.beta * truncation)
    
    def _second_moment(self, truncation: float) -> float:
        return (self.alpha * self.beta ** (self.alpha- 2.)) * incgammal(2. - self.alpha, self.beta * truncation)

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return np.exp(-self.beta * jsizes) # (n_jumps, n_samples)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        # mu = 0
        # sigma = np.sqrt(self.sigma_W2)
        # # nu = self.alpha
        # beta = self.mu_W / sigma
        # alpha = self.alpha
        # lambda_ = self.beta
        # kappa = self.alpha
        # mu = self.mu_W

        # delta = np.sqrt(alpha**2 - beta**2)
        
        # # Calculate z
        # z = (x - mu) / sigma
        
        # # Calculate the scaling constant C
        # nu = 1
        # C = alpha / (gammafnc(1 - alpha) * (lambda_)**alpha)
        
        # # Calculate the terms in the PDF formula
        # A = (2 * delta**nu) / (np.sqrt(2 * np.pi) * sigma * gammafnc(nu) * (2**(nu - 1)))
        # B = np.abs(z)**(nu - 0.5) * kv(nu - 0.5, delta * np.abs(z))
        # E = np.exp(beta * (x - mu) / sigma + (lambda_ - alpha) * nu / delta)
        
        # return C * A * B * E

        # x = np.where(x == 0.0, np.finfo(float).eps, x)
        pass