import argparse
import datetime
import numpy
import numpy.random
import random
import scipy.stats


from .make_problems import (
    make_problem,
    disjoint_union_of_models,
)

from hphmm.model import (
    HybridPoissonHMMv2 as HybridPoissonHMM,
    FixedGammaHMM,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n-problems', type=int, default=5, help='optional number of problems')
    p.add_argument('-s', '--seed', type=str, default='', help='optional seed, as hex string, e.g. 0xc0ffee')
    return p.parse_args()


def make_seedstring():
    return hex(random.getrandbits(8 * 16))


def parse_seedstring(seedstring):
    return int(seedstring, base=16)


class HPHMM:
    def __init__(self):
        pass

    def __repr__(self):
        return 'HPHMM()'

    def prepare(self, problem):
        hphmm = HybridPoissonHMM(
            transition_matrix=problem.model.transition_matrix,
            signal_matrix=problem.model.signal_matrix,
        )
        q0 = numpy.zeros((problem.model.n, 3), dtype=numpy.float64)
        q0[:, 0] = problem.state_prior
        q0[:, 1] = problem.noise_model.prior_alpha
        q0[:, 2] = problem.noise_model.prior_beta
        return hphmm, q0


class FGHMM:
    def __init__(self, r):
        self.r = r

    def __repr__(self):
        return 'FGHMM(r=%r)' % (self.r, )

    def prepare(self, problem):
        r = self.r
        base_model = problem.model
        base_n = problem.model.n

        # Approximate possible noise levels with r different pinned Gamma
        # distributions. Partition the unit interval [0, 1] into r equally
        # sized sub intervals. Pick the midpoint of each. Use these to
        # compute quantiles of the prior Gamma(alpha, beta) distribution over
        # the noise rate lambda. For the i-th lambda, define (alpha_i, beta_i)
        # as
        #   alpha_i := lambda_i * beta_i
        #   beta_i := beta + n
        # where n is the total number of observations in the dataset (the
        # number of timesteps).
        prior_alpha = problem.noise_model.prior_alpha
        prior_beta = problem.noise_model.prior_beta
        midpoints = (1.0 / r) * (0.5 + numpy.arange(r, dtype=numpy.float64))
        lambdas = scipy.stats.gamma.ppf(midpoints, a=prior_alpha,
                                        scale=(1.0 / prior_beta))
        fixed_betas = (prior_beta + problem.n_timesteps) * numpy.ones((r,), dtype=numpy.float64)
        fixed_alphas = lambdas * fixed_betas

        alpha_beta = numpy.zeros((r, 2), dtype=numpy.float64)
        alpha_beta[:, 0] = fixed_alphas
        alpha_beta[:, 1] = fixed_betas

        p0 = numpy.zeros((r, base_n), dtype=numpy.float64)
        for i in range(r):
            p0[i, :] = problem.state_prior
        z = numpy.sum(p0)
        inv_z = 1.0 / z
        p0 *= inv_z

        fghmm = FixedGammaHMM(
            transition_matrix=base_model.transition_matrix,
            signal_matrix=base_model.signal_matrix,
            alpha_beta=alpha_beta,
        )
        return fghmm, p0


class Oracle:
    def __init__(self):
        pass

    def __repr__(self):
        return 'Oracle()'

    def prepare(self, problem):
        base_model = problem.model
        base_n = problem.model.n

        # Cheat and look at the true noise level used for this problem
        # hack: assume we got many observations of exactly that rate
        r = 1 # we consider one option for noise level: the true noise level
        beta = 1.0e9
        alpha = problem.noise_model.rate * beta

        alpha_beta = numpy.zeros((r, 2), dtype=numpy.float64)
        alpha_beta[0, 0] = alpha
        alpha_beta[0, 1] = beta

        # Cheat and look at the first state in the trajectory.
        # Concentrate all mass in it
        first_true_state = problem.state_trajectory[0]
        p0 = numpy.zeros((r, base_n), dtype=numpy.float64)
        p0[0, first_true_state] = 1.0

        fghmm = FixedGammaHMM(
            transition_matrix=base_model.transition_matrix,
            signal_matrix=base_model.signal_matrix,
            alpha_beta=alpha_beta,
        )
        return fghmm, p0


def main():
    args = parse_args()
    if args.seed == '':
        print('seed not specified. generating a seed')
        seedstring = make_seedstring()
    else:
        print('using pinned seed %s' % (args.seed, ))
        seedstring = args.seed

    print('seed: %s' % (seedstring, ))

    seed = parse_seedstring(seedstring)
    rng = numpy.random.default_rng(seed)

    method_factories = [
        HPHMM(),
        FGHMM(r=1),
        FGHMM(r=5),
        FGHMM(r=10),
        FGHMM(r=20),
        Oracle(),
    ]

    n_problems = args.n_problems
    problems = {}

    print('preparing %d synthetic problems' % (n_problems, ))
    for i in range(n_problems):
        print('.', end='', flush=True)
        problems[i] = make_problem(rng)
    print('done')

    n_methods = len(method_factories)

    log_z_by_problem_method = numpy.zeros((n_problems, n_methods), dtype=numpy.float64)
    time_by_problem_method = numpy.zeros((n_problems, n_methods), dtype=numpy.float64)

    print('running competing methods on synthetic problems')
    method_j = 0
    for method_factory in method_factories:
        for problem_i in range(n_problems):
            problem = problems[problem_i]
            model, p0 = method_factory.prepare(problem)
            t0 = datetime.datetime.now()
            p, log_z = model.forward(problem.observations, p0)
            t1 = datetime.datetime.now()
            delta_t = (t1 - t0).total_seconds()
            print('method %12r\tproblem %d\tlog_z = %r\tdelta_t = %.4f' % (method_factory, problem_i, log_z, delta_t))
            log_z_by_problem_method[problem_i, method_j] = log_z
            time_by_problem_method[problem_i, method_j] = delta_t
        method_j += 1

    print('results (log-prob):')
    colheaders = '\t'.join('%12s' % (repr(f), ) for f in method_factories)
    print(colheaders)
    for i in range(n_problems):
        line = '\t'.join('%12.1f' % (log_z_by_problem_method[i, j], ) for j in range(n_methods))
        print(line)

    print('results (time):')
    colheaders = '\t'.join('%12s' % (repr(f), ) for f in method_factories)
    print(colheaders)
    for i in range(n_problems):
        line = '\t'.join('%2.4f' % (time_by_problem_method[i, j], ) for j in range(n_methods))
        print(line)


if __name__ == '__main__':
    main()
