import numpy
import numpy.random
from scipy.special import digamma
import pytest


from . import (
    expected_rate_of_gamma_mixture,
    expected_log_rate_of_gamma_mixture,
    fit_batch_gamma_dists_to_gamma_mixtures,
    fit_gamma_dist_to_gamma_mixture,
    rough_fit_batch_gamma_dists_to_gamma_mixtures,
)


@pytest.fixture(scope='module')
def rng():
    seed = 43902
    return numpy.random.default_rng(seed=seed)


@pytest.fixture(scope='module', params=["accurate", "rough"])
def rtol_and_batch_fit(request):
    if request.param == "rough":
        return 1.0e-4, rough_fit_batch_gamma_dists_to_gamma_mixtures
    if request.param == "accurate":
        return 5.0e-5, fit_batch_gamma_dists_to_gamma_mixtures


@pytest.fixture(scope='module')
def n_fuzz_trials():
    return 1


@pytest.fixture(scope='module')
def n_mixtures_per_trial():
    return 320000


def relative_error(theta, expected_rate, expected_log_rate):
    alpha_tilde, beta_tilde = theta
    tilde_rate = alpha_tilde / beta_tilde
    tilde_log_rate = digamma(alpha_tilde) - numpy.log(beta_tilde)
    return max(
        abs((expected_rate - tilde_rate) / expected_rate),
        abs((expected_log_rate - tilde_log_rate) / expected_log_rate),
    )


def make_batch_of_hard_examples():
    mix_lengths = []
    mix_cs = []
    mix_alphas = []
    mix_betas = []

    def add(c, alphas, betas):
        assert len(c) == len(alphas)
        assert len(c) == len(betas)
        assert abs(sum(c) - 1.0) <= 1.0e-7
        c /= sum(c)
        mix_cs.append(c)
        mix_alphas.append(alphas)
        mix_betas.append(betas)
        mix_lengths.append(len(c))

    c = numpy.array([
        0.17174176, 0.1717406 , 0.05142631, 0.02522303, 0.08571794,
        0.04826873, 0.051354  , 0.07576533, 0.02714939, 0.04019759,
        0.03903137, 0.02616647, 0.10315704, 0.08306041])
    alphas = numpy.array([
        9.06322469,  5.03899519, 14.3669213 ,  8.08869652,  1.47116119,
        11.74671093,  9.41258223,  0.59615235,  3.43848816, 11.80365131,
        11.92002805, 15.94657463, 16.08818745, 16.26530307])
    betas = numpy.array([
        5.33368462, 7.65490605, 7.07054815, 3.53273093, 3.6450181 ,
        7.67670211, 7.42729331, 8.08019977, 1.77404162, 9.19694889,
        5.06233783, 2.55345075, 9.99140258, 2.14951783])
    add(c, alphas, betas)

    c = numpy.array([
        0.2095968 , 0.24319593, 0.05657532, 0.23345548, 0.09776819,
        0.15940828])
    alphas = numpy.array([
        4.10293788,  1.01046237, 16.82295376, 13.53712183, 14.41873546,
        6.34783553])
    betas = numpy.array([
        2.34196591, 5.59797794, 4.53506847, 7.91390396, 2.88620039,
        2.73683676])
    add(c, alphas, betas)

    c = numpy.array([
        0.06739026, 0.07165743, 0.09403369, 0.03824476, 0.09027672,
        0.10638134, 0.09519034, 0.01494345, 0.11091705, 0.07683023,
        0.03434097, 0.05836284, 0.07704543, 0.06438548])
    alphas = numpy.array([
        2.29491824,  3.71687646,  2.5112914 , 19.64030107,  1.57324705,
        13.92049439,  5.98595523, 10.19055963, 14.41649833,  0.36558322,
        11.48828498, 15.42490588,  8.30820493, 12.0384796 ])
    betas = numpy.array([
        0.53688768, 3.62828329, 6.8676808 , 8.50143399, 4.86741769,
        7.57817314, 5.47948149, 1.25027835, 2.25321465, 2.67774143,
        9.29424004, 6.40877878, 4.64244398, 5.49939609])
    add(c, alphas, betas)

    c = numpy.array([
        0.06094906, 0.05455422, 0.04226409, 0.07360946, 0.03984035,
        0.05415634, 0.06675147, 0.08047874, 0.04015222, 0.06295773,
        0.06394108, 0.09164933, 0.04542431, 0.03516104, 0.0896529 ,
        0.02302977, 0.04854092, 0.00133156, 0.02555539])
    alphas = numpy.array([
        3.63249593,  0.78359292,  0.49937274,  4.82604271, 15.0275789 ,
        7.28643421,  2.81594973, 18.63161914, 16.36763414,  1.71278158,
        1.6671194 , 17.54545838,  6.81479005,  8.83169485,  4.32236396,
         3.26989195,  0.81997786, 17.91911166,  3.24554951])
    betas = numpy.array([
        1.53354607, 1.1572384 , 4.62239583, 2.18889165, 6.4006518 ,
        5.17070604, 5.50105955, 3.19853415, 9.2715749 , 2.60384866,
        8.22936357, 2.51693339, 1.82032835, 1.94058701, 2.66441025,
        6.74642501, 7.04973338, 1.97330448, 7.10373949])
    add(c, alphas, betas)

    c = numpy.array([
        0.12799161, 0.07970121, 0.00112451, 0.10347689, 0.06990676,
        0.07574694, 0.03649311, 0.07076179, 0.13608737, 0.15453556,
        0.13559676, 0.00857749])
    alphas = numpy.array([
        7.63535937, 10.30918286,  2.97344193,  9.0494593 ,  1.07591431,
        12.11305228,  0.85500947,  3.12482748,  6.0724857 ,  3.49222919,
        11.63912565, 11.38301799])
    betas = numpy.array([
        4.98623433, 5.11143794, 5.15706283, 6.8024076 , 2.40030211,
        6.29506446, 2.78755001, 4.80909195, 4.78727093, 4.92318737,
        5.84801524, 6.32157057])
    add(c, alphas, betas)

    c = numpy.array([0.38354495, 0.36417459, 0.25228047])
    alphas = numpy.array([ 3.81453454, 14.25942937,  0.65067866])
    betas = numpy.array([1.79128631, 5.07242982, 2.75626998])
    add(c, alphas, betas)

    c = numpy.array([0.32843092, 0.06083425, 0.27918106, 0.33155377])
    alphas = numpy.array([16.59223925,  4.97030335,  0.76911118,  8.15268122])
    betas = numpy.array([2.930591  , 7.21334564, 3.83106814, 5.10559445])
    add(c, alphas, betas)

    c = numpy.array([
        0.04979693, 0.08779501, 0.0950741 , 0.03831677, 0.09039928,
        0.08514387, 0.06387562, 0.08687208, 0.06115016, 0.09455369,
        0.05939446, 0.08677384, 0.0885959 , 0.01225828])
    alphas = numpy.array([
        11.57416547,  6.91308957, 14.07252542,  8.71790397,  7.2998117 ,
        1.44288037,  6.54783741,  2.40778924,  0.70538   , 12.37370666,
        11.61799947,  6.61803241,  4.05614527, 18.29718043])
    betas = numpy.array([
        9.36255311, 4.3524829 , 5.89680925, 0.42941463, 7.13353454,
        1.9110169 , 4.35014579, 1.77901889, 9.86758063, 7.46189585,
        3.83586981, 8.4862775 , 9.12434376, 4.86092547])
    add(c, alphas, betas)

    # Nasty example. Increasing number of Halley's method iterations
    # doesn't appear to help.
    c = numpy.array([
        0.18569141, 0.12771625, 0.0835672, 0.1340494, 0.20193201,
        0.14130824, 0.05544657, 0.07028893])
    alphas = numpy.array([
        1.16340625e+00, 6.56767644e+03, 3.44695157e+03, 1.77372732e-01,
        4.34324328e+03, 1.93266757e+01, 1.60593812e+00, 1.19390716e+5])
    betas = numpy.array([
        2.04918167e+01, 2.31999333e+03, 5.67541392e+03, 1.72020779e+00,
        5.21686963e+00, 1.27125810e+01, 1.58845935e+02, 2.81032632e+03])
    add(c, alphas, betas)

    lengths = numpy.asarray(mix_lengths, dtype=numpy.int64)
    n_components = lengths.sum()

    cab = numpy.empty(shape=(n_components, 3), dtype=numpy.float64)
    cab[:, 0] = numpy.concatenate(mix_cs, dtype=numpy.float64)
    cab[:, 1] = numpy.concatenate(mix_alphas, dtype=numpy.float64)
    cab[:, 2] = numpy.concatenate(mix_betas, dtype=numpy.float64)
    return (lengths, cab)


def make_batch_problems(rng, n_mixtures):
    min_m = 1
    max_m = 20

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

        # Generate alphas and betas in terms of rate
        # and sample size
        rate = 10.0 ** rng.uniform(-3.0, 3.0, size=m)
        samples = 10.0 ** rng.uniform(-2.0, 5.0, size=m)

        # C.f. https://en.wikipedia.org/wiki/Poisson_distribution#Bayesian_inference
        alphas = rate * samples
        betas = samples

        assert numpy.all(alphas > 0)
        assert numpy.all(betas > 0)

        mix_lengths.append(m)
        mix_cs.append(c)
        mix_alphas.append(alphas)
        mix_betas.append(betas)

    lengths = numpy.asarray(mix_lengths, dtype=numpy.int64)
    n_components = lengths.sum()

    cab = numpy.empty(shape=(n_components, 3), dtype=numpy.float64)
    cab[:, 0] = numpy.concatenate(mix_cs, dtype=numpy.float64)
    cab[:, 1] = numpy.concatenate(mix_alphas, dtype=numpy.float64)
    cab[:, 2] = numpy.concatenate(mix_betas, dtype=numpy.float64)
    return (lengths, cab)


def test_batch_fit_on_hard_examples(rtol_and_batch_fit):
    rtol, batch_fit = rtol_and_batch_fit

    mix_lengths, mix_cab = make_batch_of_hard_examples()
    n_mixtures = len(mix_lengths)

    acc_iters = 0
    acc_mixtures = 0

    rel_errors = numpy.zeros((n_mixtures, ), dtype=numpy.float64)
    out_alphas = numpy.zeros((n_mixtures, ), dtype=numpy.float64)
    out_betas = numpy.zeros((n_mixtures,), dtype=numpy.float64)

    result = batch_fit(
        mix_lengths,
        mix_cab,
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
        c = mix_cab[start:end, 0]
        alphas = mix_cab[start:end, 1]
        betas = mix_cab[start:end, 2]
        expected_rate = expected_rate_of_gamma_mixture(c, alphas, betas)
        expected_log_rate = expected_log_rate_of_gamma_mixture(c, alphas, betas)

        # Extract parameters of single Gamma fit
        alpha_star = out_alphas[mix_i]
        beta_star = out_betas[mix_i]
        theta_star = numpy.asarray([alpha_star, beta_star])

        # Measure relative approximation error of expected raet & expected log rate.
        rel_errors[mix_i] = relative_error(theta_star, expected_rate, expected_log_rate)
        start = start + mix_lengths[mix_i]

    print('total iters %r' % (acc_iters, ))
    print('mean iters per mix %r' % (acc_iters / acc_mixtures, ))

    max_rel_error = numpy.amax(rel_errors)
    mean_rel_error = numpy.mean(rel_errors)

    print('max_rel_error %r' % (max_rel_error,))
    print('mean_rel_error per mix %r' % (mean_rel_error,))

    assert max_rel_error <= rtol


def test_fuzztest_mixture_fitting(rng, n_fuzz_trials, n_mixtures_per_trial, rtol_and_batch_fit):
    rtol, batch_fit = rtol_and_batch_fit

    n_trials = n_fuzz_trials
    n_mixtures = n_mixtures_per_trial

    acc_iters = 0
    acc_mixtures = 0

    rel_errors = numpy.zeros((n_trials, n_mixtures, ), dtype=numpy.float64)

    for trial_i in range(n_trials):

        mix_lengths, mix_cab = make_batch_problems(rng, n_mixtures)

        out_alphas = numpy.zeros((n_mixtures, ), dtype=numpy.float64)
        out_betas = numpy.zeros((n_mixtures,), dtype=numpy.float64)

        result = batch_fit(
            mix_lengths,
            mix_cab,
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
            c = mix_cab[start:end, 0]
            alphas = mix_cab[start:end, 1]
            betas = mix_cab[start:end, 2]
            expected_rate = expected_rate_of_gamma_mixture(c, alphas, betas)
            expected_log_rate = expected_log_rate_of_gamma_mixture(c, alphas, betas)

            # Extract parameters of single Gamma fit
            alpha_star = out_alphas[mix_i]
            beta_star = out_betas[mix_i]
            theta_star = numpy.asarray([alpha_star, beta_star])

            # Measure relative approximation error of expected raet & expected log rate.
            rel_errors[trial_i, mix_i] = relative_error(theta_star, expected_rate, expected_log_rate)

            start = start + mix_lengths[mix_i]

        worst_trial_i = numpy.argmax(rel_errors[trial_i, :])

        if rel_errors[trial_i, worst_trial_i] > rtol:
            print('trial %d mix %d example with relerror %r > rtol %r' % (
                trial_i,
                worst_trial_i,
                rel_errors[trial_i, worst_trial_i],
                rtol,
            ))
            start_worst = mix_lengths[:worst_trial_i].sum()
            end_worst = start_worst + mix_lengths[worst_trial_i]

            c = mix_cab[start_worst:end_worst, 0]
            alphas = mix_cab[start_worst:end_worst, 1]
            betas = mix_cab[start_worst:end_worst, 2]

            print('c = %r' % (c, ))
            print('alphas = %r' % (alphas,))
            print('betas = %r' % (betas,))


    print('total iters %r' % (acc_iters, ))
    print('mean iters per mix %r' % (acc_iters / acc_mixtures, ))

    max_rel_error = numpy.amax(rel_errors)
    mean_rel_error = numpy.mean(rel_errors)

    print('max_rel_error %r' % (max_rel_error,))
    print('mean_rel_error per mix %r' % (mean_rel_error,))

    assert max_rel_error <= rtol

