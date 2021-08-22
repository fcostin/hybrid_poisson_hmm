import numpy
from scipy.stats import poisson

from .model import HybridPoissonHMMv2 as HybridPoissonHMM


def main():
    print('preparing')
    n = 1000 # number of states in state space
    n_transitions_per_state = 5

    indices = numpy.arange(n)
    transition_matrix = numpy.zeros((n, n), dtype=numpy.float64)
    for i in range(n):
        js = numpy.random.choice(indices, size=n_transitions_per_state, replace=False)
        js = numpy.sort(js)
        cs = numpy.random.uniform(0.0, 1.0, size=n_transitions_per_state)
        cs /= numpy.sum(cs)
        assert numpy.isclose(1.0, numpy.sum(cs))
        transition_matrix[i, js] = cs

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
    one_nth = 1.0 / n
    for i in range(n):
        q0[i, 0] = one_nth
        q0[i, 1] = 1.0 # ??? want weak prior over lambda
        q0[i, 2] = 0.5 # ??? want weak prior over lambda

    # mock up some synthetic data
    n_times = 500
    signals = numpy.asarray(numpy.random.uniform(0.0, 1.0, n_times) < rho[2], dtype=int)
    noise_rate = 0.01
    noise = poisson.rvs(noise_rate, size=n_times)
    observations = signals + noise

    print('ready')
    for i in range(100):
        hphmm.forward(observations, q0)
        print('.',end='', flush=True)
    print('done')

if __name__ == '__main__':
    main()
