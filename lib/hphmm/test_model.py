import typing
import numpy
import numpy.random
import numpy.typing
from scipy.stats import poisson
import pytest

from .model import HybridPoissonHMM

class ModelParameters(typing.NamedTuple):
    transition_matrix: numpy.typing.NDArray
    signal_matrix: numpy.typing.NDArray


@pytest.fixture(scope='module')
def pinned_rng():
    # scipy stats uses numpy.random directly.
    numpy.random.seed(seed=43902) # mutates global state.
    return None


def make_demo_model_params(n: int, delta: float, rho_min: float, rho_max: float):
    """
    :param n: number of hidden states
    :param delta: probability of transition from s_i to s_j , j != i
    :param rho_min: emission probability for least active state (state 0)
    :param rho_max:  emission probability for most active state (state n-1)
    :return:
    """
    assert n > 0
    assert 0.0 <= delta
    assert delta <= 1.0
    assert 0.0 <= rho_min
    assert rho_min <= rho_max
    assert rho_max <= 1.0

    diag = numpy.eye(n, dtype=numpy.float64)
    if n > 1:
        transition_matrix = (delta / (n - 1.0)) * (
                    numpy.ones((n, n), dtype=numpy.float64) - diag) + (1.0 - delta) * diag
    else:
        transition_matrix = numpy.eye(n, dtype=numpy.float64)

    rho = numpy.linspace(rho_min, rho_max, n, dtype=numpy.float64)
    assert rho.shape == (n,)
    max_k = 1
    signal_matrix = numpy.zeros((max_k + 1, n))
    signal_matrix[1, :] = rho
    signal_matrix[0, :] = 1.0 - rho

    return ModelParameters(
        transition_matrix=transition_matrix,
        signal_matrix=signal_matrix,
    )


def make_prior(n: int, alpha: float, beta: float):
    assert n > 0
    assert alpha > 0.0
    assert beta > 0.0
    q0 = numpy.empty((n, 3), dtype=numpy.float64)
    for i in range(n):
        # Uniform prior for P(state)
        q0[i, 0] = 1.0 / n
        # Gamma distribution for P(lambda | state)
        q0[i, 1] = alpha
        q0[i, 2] = beta
    return q0


def make_synthetic_observations(n_times: int, rho: float, noise_rate: float):
    signals = numpy.asarray(numpy.random.uniform(0.0, 1.0, n_times) < rho, dtype=int)
    noise = poisson.rvs(noise_rate, size=n_times)
    observations = signals + noise
    return observations


def test_single_state_noise_only_model_updates_gamma_posterior(pinned_rng):
    """
    Sanity check that noise-only model updates Gamma distribution according
    to normal rules for Poisson rate under a Gamma conjugate prior.

    Hidden Markov model has 1 state, doesnt emit any signal.
    Any event counts in observation are entirely due to Poisson noise.
    In the approximate posterior distribution
    P(lambda, s | y_{1:t}) approx q(lambda, s | y_{1:t})
    we have - by assumption - the decomposition
    q(lambda, s | y_{1:t}) = q(lambda | s, y_{1:t}) q(s | y_{1:t})
    where the first factor q(lambda | s, y_{1:t}) is a Gamma distribution.

    If the prior for q was q0(lambda, s) = q0(lambda | s) q0(s)
    where q0(lambda | s) = Gamma(lambda ; alpha_0, beta_0)
    then we expect the final posterior factor q(lambda | s, y_{1:t}) to be

    q(lambda | s, y_{1:t}) = Gamma(lambda ; alpha_0 + sum_t y_t , beta_0 + T)

    where
        T is the number of observations and
        sum_{t=1}^Y y_t is the total observed event count.
    """
    n_states = 1
    params = make_demo_model_params(
        n=n_states,
        delta=0.0,
        rho_min=0.0,
        rho_max=0.0,
    )
    prior_alpha = 1.0
    prior_beta = 0.5
    q0 = make_prior(n_states, alpha=prior_alpha, beta=prior_beta)

    observations = make_synthetic_observations(n_times=100, rho=0.0, noise_rate=1.0)

    n_observations = len(observations)
    net_event_count = numpy.sum(observations)

    expected_alpha = prior_alpha + net_event_count
    expected_beta = prior_beta + n_observations

    model = HybridPoissonHMM(
        transition_matrix=params.transition_matrix,
        signal_matrix=params.signal_matrix,
    )

    q = model.forward(observations, q0)

    final_c = q[0, 0]
    final_alpha = q[0, 1]
    final_beta = q[0, 2]

    assert (0.0 < final_c and
            numpy.isclose(expected_alpha, final_alpha) and
            numpy.isclose(expected_beta, final_beta))


def test_two_independent_state_noise_only_model_updates_gamma_posterior(pinned_rng):
    """
    Trivial two-state model - states cannot transition or emit event counts.

    Check that the distribution of the noise rate lambda given the state are
    equal and exactly match what we expect for updating parameters of Gamma
    conjugate prior distribution for observations assumed to be generated by a
    Poisson distribution.
    """
    n_states = 2
    params = make_demo_model_params(
        n=n_states,
        delta=0.0,   # no transitions between states permitted
        rho_min=0.0,
        rho_max=0.0,
    )
    prior_alpha = 1.0
    prior_beta = 0.5
    q0 = make_prior(n_states, alpha=prior_alpha, beta=prior_beta)

    observations = make_synthetic_observations(n_times=100, rho=0.0, noise_rate=1.0)

    n_observations = len(observations)
    net_event_count = numpy.sum(observations)

    expected_alpha = prior_alpha + net_event_count
    expected_beta = prior_beta + n_observations

    model = HybridPoissonHMM(
        transition_matrix=params.transition_matrix,
        signal_matrix=params.signal_matrix,
    )

    q = model.forward(observations, q0)

    final_c_state_0 = q[0, 0]
    final_alpha_state_0 = q[0, 1]
    final_beta_state_0 = q[0, 2]
    final_c_state_1 = q[1, 0]
    final_alpha_state_1 = q[1, 1]
    final_beta_state_1 = q[1, 2]

    assert (0.0 < final_c_state_0 and
            0.0 < final_c_state_1 and
            numpy.isclose(final_c_state_0, final_c_state_1) and
            numpy.isclose(expected_alpha, final_alpha_state_0) and
            numpy.isclose(expected_beta, final_beta_state_0) and
            numpy.isclose(expected_alpha, final_alpha_state_1) and
            numpy.isclose(expected_beta, final_beta_state_1))


def test_single_state_mixed_noise_and_signal_updates_gamma_posterior(pinned_rng):
    """
    """
    n_states = 1
    params = make_demo_model_params(
        n=n_states,
        delta=0.0,
        rho_min=0.5,
        rho_max=0.5,
    )
    prior_alpha = 1.0
    prior_beta = 0.5
    q0 = make_prior(n_states, alpha=prior_alpha, beta=prior_beta)

    observations = make_synthetic_observations(n_times=100, rho=0.0, noise_rate=1.0)

    n_observations = len(observations)
    net_event_count = numpy.sum(observations)
    # Signal model generates at most 1 event count per observation
    # hence if we see an observation with 4 event counts, we know that at least
    # three of them must be noise.
    net_count_of_events_that_cannot_possibly_be_signal = numpy.sum(numpy.maximum(0, observations-1))

    expected_alpha_min = prior_alpha + net_count_of_events_that_cannot_possibly_be_signal
    expected_alpha_max = prior_alpha + net_event_count
    # expected_beta_without_compression = prior_beta + n_observations

    model = HybridPoissonHMM(
        transition_matrix=params.transition_matrix,
        signal_matrix=params.signal_matrix,
    )

    q = model.forward(observations, q0)

    final_c = q[0, 0]
    final_alpha = q[0, 1]
    final_beta = q[0, 2]

    # Not sure if we can expect bounds on alpha since we are projecting
    # mixtures of Gamma distributions back to a single Gamma distribution.
    # Perhaps we might be able to prove bounds on the expected rate and
    # expected log rate of the result vs the expected rate and log rate of a
    # theoretical uncompressed mixture of Gammas.
    assert (0.0 < final_c and
            (expected_alpha_min <= final_alpha) and
            (final_alpha <= expected_alpha_max))


def test_two_independent_state_mixed_noise_and_signal_can_infer_most_probable_state(pinned_rng):
    """
    Simple two state model. No transitions permitted between states. Each state
    emits at most one event count with distinct probabilities.
    """
    n_states = 2
    params = make_demo_model_params(
        n=n_states,
        delta=0.0,   # no transitions between states permitted
        rho_min=0.2,
        rho_max=0.8,
    )
    prior_alpha = 1.0
    prior_beta = 0.5
    q0 = make_prior(n_states, alpha=prior_alpha, beta=prior_beta)

    observations = make_synthetic_observations(n_times=100, rho=0.75, noise_rate=0.05)

    n_observations = len(observations)
    net_event_count = numpy.sum(observations)

    expected_alpha = prior_alpha + net_event_count
    expected_beta = prior_beta + n_observations

    model = HybridPoissonHMM(
        transition_matrix=params.transition_matrix,
        signal_matrix=params.signal_matrix,
    )

    q = model.forward(observations, q0)

    final_c_state_0 = q[0, 0]
    final_c_state_1 = q[1, 0]

    # TODO prove this bound (hopefully tighter).
    epsilon = 100.0
    assert (0.0 < final_c_state_0 and
            0.0 < final_c_state_1 and
            final_c_state_0 < epsilon * final_c_state_1)