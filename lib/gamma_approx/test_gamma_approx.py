import numpy
import numpy.random
from scipy.special import digamma
import pytest


from . import (
    expected_rate_of_gamma_mixture,
    expected_log_rate_of_gamma_mixture,
    fit_batch_gamma_dists_to_gamma_mixtures,
    fit_gamma_dist_to_gamma_mixture,
)


@pytest.fixture(scope='module')
def rng():
    seed = 123
    return numpy.random.default_rng(seed=seed)


def make_batch_problems(rng, n_mixtures):
    min_m = 3
    max_m = 9

    mix_lengths = []
    mix_cs = []
    mix_alphas = []
    mix_betas = []
    for mixture_i in range(n_mixtures):
        # Sample number of Gammas in the mixture
        m = rng.integers(min_m, max_m + 1)
        # Sample convex combination
        c = rng.uniform(0.0, 1.0, size=m)
        c /= numpy.sum(c)

        # Sample alphas and betas
        alphas = rng.uniform(0.01, 20.0, size=m)
        betas = rng.uniform(0.01, 10.0, size=m)

        assert numpy.all(alphas > 0)
        assert numpy.all(betas > 0)

        mix_lengths.append(m)
        mix_cs.append(c)
        mix_alphas.append(alphas)
        mix_betas.append(betas)

    lengths = numpy.asarray(mix_lengths, dtype=numpy.int64)
    c = numpy.concatenate(mix_cs, dtype=numpy.float64)
    alphas = numpy.concatenate(mix_alphas, dtype=numpy.float64)
    betas = numpy.concatenate(mix_betas, dtype=numpy.float64)

    return (lengths, c, alphas, betas)


def test_fuzztest_mixture_fitting(rng):

    n_trials = 100
    n_mixtures = 800

    acc_iters = 0
    acc_mixtures = 0

    rel_errors = numpy.zeros((n_trials, n_mixtures, ), dtype=numpy.float64)

    for trial_i in range(n_trials):

        mix_lengths, mix_cs, mix_alphas, mix_betas = make_batch_problems(rng, n_mixtures)

        out_alphas = numpy.zeros((n_mixtures, ), dtype=numpy.float64)
        out_betas = numpy.zeros((n_mixtures,), dtype=numpy.float64)

        result = fit_batch_gamma_dists_to_gamma_mixtures(
            mix_lengths,
            mix_cs,
            mix_alphas,
            mix_betas,
            out_alphas,
            out_betas,
        )
        assert result['status'] == 0, repr(result)

        acc_iters += result['iters']
        acc_mixtures += n_mixtures

        start = 0
        end = 0
        for mix_i in range(n_mixtures):
            end = end + mix_lengths[mix_i]

            # Compute expected rate and expected log rate of mixture
            c = mix_cs[start:end]
            alphas = mix_alphas[start:end]
            betas = mix_betas[start:end]
            expected_rate = expected_rate_of_gamma_mixture(c, alphas, betas)
            expected_log_rate = expected_log_rate_of_gamma_mixture(c, alphas, betas)

            # Extract parameters of single Gamma fit
            alpha_star = out_alphas[mix_i]
            beta_star = out_betas[mix_i]
            theta_star = numpy.asarray([alpha_star, beta_star])

            # Measure relative approximation error of expected raet & expected log rate.
            rel_errors[trial_i, mix_i] = relative_error(theta_star, expected_rate, expected_log_rate)

            start = start + mix_lengths[mix_i]

    print('total iters %r' % (acc_iters, ))
    print('mean iters per mix %r' % (acc_iters / acc_mixtures, ))

    max_rel_error = numpy.amax(rel_errors)
    mean_rel_error = numpy.mean(rel_errors)

    print('max_rel_error %r' % (max_rel_error,))
    print('mean_rel_error per mix %r' % (mean_rel_error,))

    assert max_rel_error <= 1.0e-5


def relative_error(theta, expected_rate, expected_log_rate):
    alpha_tilde, beta_tilde = theta
    tilde_rate = alpha_tilde / beta_tilde
    tilde_log_rate = digamma(alpha_tilde) - numpy.log(beta_tilde)
    return max(
        abs((expected_rate - tilde_rate) / expected_rate),
        abs((expected_log_rate - tilde_log_rate) / expected_log_rate),
    )
