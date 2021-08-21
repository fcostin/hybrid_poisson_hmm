"""
Simple four state demo

states space S = {s0, s1, s2, s3}

Transition model:

Fix delta in (0, 1).

P(X_{t+1} = s' | X_t = s) =     {   1 - delta       if s' = s
                                {       delta / 3   otherwise

Signal model

For each state s in S
Define rho_s in (0, 1).

P(Z_t = k | X_t = s)    :=  {   rho_s       if k = 1
                            {   1 - rho_s   if k = 0
                            {   0           otherwise

Noise model

K_t ~ Poisson(lambda), lambda ~ Gamma(alpha_0, beta_0)

Observation model

Y_t := Z_t + K_t

"""

import numpy
from scipy.stats import poisson

from .model import HybridPoissonHMMv2 as HybridPoissonHMM


def main():
    n = 100 # number of states in state space

    delta = 1.0e-4
    diag = numpy.eye(n, dtype=numpy.float64)
    transition_matrix = (delta / (n - 1.0)) * (numpy.ones((n, n), dtype=numpy.float64) - diag) + (1.0 - delta) * diag

    rho = numpy.geomspace(0.1, 0.9, n, dtype=numpy.float64)
    assert rho.shape == (n, )
    max_k = 1
    signal_matrix = numpy.zeros((max_k + 1, n))
    signal_matrix[1, :] = rho
    signal_matrix[0, :] = 1.0 - rho

    hphmm = HybridPoissonHMM(
        transition_matrix=transition_matrix,
        signal_matrix=signal_matrix,
    )

    q0 = numpy.empty((n, 3), dtype=numpy.float64)
    for i in range(n):
        q0[i, 0] = 1.0 / n
        q0[i, 1] = 1.0 # ??? want weak prior over lambda
        q0[i, 2] = 0.5 # ??? want weak prior over lambda

    # mock up some synthetic data
    n_times = 10000
    signals = numpy.asarray(numpy.random.uniform(0.0, 1.0, n_times) < rho[2], dtype=int)
    noise_rate = 0.001
    noise = poisson.rvs(noise_rate, size=n_times)
    observations = signals + noise

    # infer
    q = hphmm.forward(observations, q0)


if __name__ == '__main__':
    main()