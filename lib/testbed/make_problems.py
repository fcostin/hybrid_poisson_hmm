import numpy
import numpy.typing
import typing
import scipy.stats

from hphmm.model import (CSRMatrix, make_csr_matrix_from_dense)


class Model(typing.NamedTuple):
    n: int # number of states
    transition_matrix: CSRMatrix # shape (n, n) compressed sparse row
    signal_matrix: numpy.typing.NDArray[numpy.float64] # shape (k, n)


class PoissonNoiseModel(typing.NamedTuple):
    rate: float
    prior_alpha: float
    prior_beta: float


class Problem(typing.NamedTuple):
    n_timesteps: int
    model: Model
    noise_model: PoissonNoiseModel
    state_prior: numpy.typing.NDArray[numpy.float64]
    state_trajectory: numpy.typing.NDArray[int]
    signal: numpy.typing.NDArray[int]
    noise: numpy.typing.NDArray[int]
    observations: numpy.typing.NDArray[int]


def uniform_integer(rng, low, high, size=None):
    # sample an integer from low to high, both endpoints inclusive
    return rng.integers(low=low, high=high+1, size=size)


def disjoint_union_of_csr_matrices(n: int, offsets: int, sub_matrices: typing.Sequence[CSRMatrix]) -> CSRMatrix:
    nonzeros = numpy.sum((len(a.data) for a in sub_matrices))

    indptr = numpy.zeros(shape=(n + 1, ), dtype=numpy.int64)
    cols = numpy.zeros(shape=(nonzeros, ), dtype=numpy.int64)
    data = numpy.zeros(shape=(nonzeros, ), dtype=numpy.float64)

    def index_global_from_local(sub_model, i):
        # i : scalar index or ndarray of indices (local to sub model)
        return offsets[sub_model] + i

    indptr[0] = 0
    k = 0
    i = 1
    for a_i, a in enumerate(sub_matrices):
        n_i = len(a.indptr)-1
        n_data = len(a.data)
        indptr[i:i+n_i] = a.indptr[1:] + k
        cols[k:k+n_data] = index_global_from_local(a_i, a.cols)
        data[k:k+n_data] = a.data
        i += n_i
        k += n_data

    assert i == len(indptr)
    assert k == nonzeros

    return CSRMatrix(
        indptr=indptr,
        cols=cols,
        data=data,
    )


def disjoint_union_of_models(sub_models: typing.Sequence[Model]) -> Model:
    m = len(sub_models)
    sizes = numpy.asarray([subm.n for subm in sub_models], dtype=int)
    n = sum(sizes)

    offsets = numpy.zeros((m, ), dtype=int)
    offsets[1:] = numpy.add.accumulate(sizes)[:-1]

    def index_global_from_local(sub_model, i):
        # i : scalar index or ndarray of indices (local to sub model)
        return offsets[sub_model] + i

    # Assemble sparse transition matrix
    transition_matrix = disjoint_union_of_csr_matrices(
        n=n,
        offsets=offsets,
        sub_matrices=[subm.transition_matrix for subm in sub_models],
    )

    # Assemble dense signal matrix
    k_max = numpy.amax([subm.signal_matrix.shape[0] for subm in sub_models])
    signal_matrix = numpy.zeros((k_max, n), dtype=numpy.float64)
    for sub_model_i, sub_model in enumerate(sub_models):
        assert sub_model.signal_matrix.shape[1] == sizes[sub_model_i]
        for i in range(sizes[sub_model_i]):
            assert numpy.all(sub_model.signal_matrix[:, i] >= 0.0)
            assert numpy.all(sub_model.signal_matrix[:, i] <= 1.0)
            assert numpy.isclose(1.0, numpy.sum(sub_model.signal_matrix[:, i]))
        k = sub_model.signal_matrix.shape[0]
        start = index_global_from_local(sub_model_i, 0)
        end = index_global_from_local(sub_model_i, sub_model.signal_matrix.shape[1])
        signal_matrix[:k, start:end] = sub_model.signal_matrix

    # Sanity check signal matrix
    for i in range(n):
        assert numpy.all(signal_matrix[:, i] >= 0.0)
        assert numpy.all(signal_matrix[:, i] <= 1.0)
        assert numpy.isclose(1.0, numpy.sum(signal_matrix[:, i]))
    result = Model(
        n=n,
        transition_matrix=transition_matrix,
        signal_matrix=signal_matrix,
    )
    return result


def make_model(rng, n_submodels):
    sub_models = []
    for sub_model_i in range(n_submodels):
        # Make the i-th sub-model

        # Define number of hidden states n
        # Label hidden states x_0, ..., x_{n-1}
        n = uniform_integer(rng, 5, 10)

        # Define transition matrix. Allow transitions in a loop.
        # n undirected edges, edge i between x_i and x_{i+1 mod n}.
        # Replace each undirected edge with 1 or 2 directed edges.
        # We only allow "fwd" or "fwd & rev", not "rev" alone.

        fwd_weight = rng.uniform(0.0, 1.0, size=n)

        pr_rev_edge = 0.5
        rev_edge = rng.uniform(0.0, 1.0, size=n) < pr_rev_edge
        rev_weight = rng.uniform(0.0, 1.0, size=n)
        rev_weight = numpy.where(rev_edge, rev_weight, 0.0)

        self_weight = rng.uniform(0.0, 1.0, size=n)

        z = fwd_weight + rev_weight + self_weight
        fwd_weight /= z
        rev_weight /= z
        self_weight /= z

        transitions = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            transitions[i, i] = self_weight[i]
            transitions[(i+1)%n, i] = fwd_weight[i]
            transitions[(i-1)%n, i] = rev_weight[i]

        for i in range(n):
            assert numpy.isclose(1.0, numpy.sum(transitions[:, i]))

        # Define signal matrix.

        # For each state, select a signalling model:
        # -- sample K in 0 ... 5
        # -- emit {0, ..., K} counts based on categorical distribution.
        # -- sample convex weights for the categorical distribution
        max_count = 6

        signal_matrix = numpy.zeros((max_count, n), dtype=numpy.float64)
        for i in range(n):
            k = uniform_integer(rng, 1, max_count)
            weights = rng.uniform(0.0, 1.0, size=k)
            weights /= numpy.sum(weights)
            signal_matrix[:k, i] = weights

        for i in range(n):
            assert numpy.all(signal_matrix[:, i] >= 0.0)
            assert numpy.all(signal_matrix[:, i] <= 1.0)
            assert numpy.isclose(1.0, numpy.sum(signal_matrix[:, i]))

        sub_models.append(
            Model(
                n=n,
                transition_matrix=make_csr_matrix_from_dense(transitions),
                signal_matrix=signal_matrix,
            )
        )
    return disjoint_union_of_models(sub_models)


def make_problem(rng):
    n_timesteps = uniform_integer(rng, 100, 500)

    n_submodels = uniform_integer(rng, 5, 50)
    model = make_model(rng, n_submodels)

    state_prior = (1.0 / model.n) * numpy.ones((model.n,), dtype=numpy.float64)

    # Choose parameters for a Gamma distribution that'll
    # probably give some reasonable rate for Poisson noise

    r = 10.0 ** rng.uniform(-1.0, 1.0) # expected noise rate
    beta = 10.0 ** rng.uniform(-1.0, 1.0) # fairly vague.
    alpha = r * beta
    # Sample the noise rate
    noise_rate = scipy.stats.gamma.rvs(a=alpha, scale=(1.0/beta), random_state=rng)

    noise_model = PoissonNoiseModel(
        rate=noise_rate,
        prior_alpha=alpha,
        prior_beta=beta,
    )

    noise = scipy.stats.poisson.rvs(noise_rate, size=n_timesteps, random_state=rng)
    signal = numpy.zeros((n_timesteps,), dtype=int)
    state_trajectory = numpy.zeros((n_timesteps+1, ), dtype=int)

    states = numpy.arange(model.n)
    state = rng.choice(states, p=state_prior)
    tr_matrix = model.transition_matrix
    s_matrix = model.signal_matrix
    ks = numpy.arange(s_matrix.shape[0])
    for t in range(n_timesteps):
        # Sample state
        state_trajectory[t] = state
        # State transition
        # FIXME CSR is the completely wrong structure for sampling. We want CSC
        next_states = []
        transition_probabilities = []
        for i in range(model.n):
            start = tr_matrix.indptr[i]
            end = tr_matrix.indptr[i + 1]
            for j in range(start, end):
                if tr_matrix.cols[j] == state:
                    next_states.append(i)
                    transition_probabilities.append(tr_matrix.data[j])
        state_prime = rng.choice(next_states, p=transition_probabilities)
        # Sample signal
        signal[t] = rng.choice(ks, p=s_matrix[:, state])
        # Advance
        state = state_prime
    state_trajectory[n_timesteps] = state

    observations = signal + noise

    return Problem(
        n_timesteps=n_timesteps,
        model=model,
        noise_model=noise_model,
        state_prior=state_prior,
        state_trajectory=state_trajectory,
        signal=signal,
        noise=noise,
        observations=observations,
    )
