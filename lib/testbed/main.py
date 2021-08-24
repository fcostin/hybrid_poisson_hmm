import argparse
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
    p.add_argument('-s', '--seed', type=str, default='', help='optional seed, as hex string, e.g. 0xc0ffee')
    return p.parse_args()


def make_seedstring():
    return hex(random.getrandbits(8 * 16))


def parse_seedstring(seedstring):
    return int(seedstring, base=16)


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

    n_problems = 100
    problem = {}

    print('preparing synthetic problems')
    for i in range(n_problems):
        print('.', end='', flush=True)
        problem[i] = make_problem(rng)
    print('done')

    fghmm_rs = (1, 5, 10) # 10, 20)

    method_names = ['hphmm'] + ['fghmm/%d' % (r, ) for r in fghmm_rs]
    n_methods = 1 + len(fghmm_rs)
    log_z_by_problem_method = numpy.zeros((n_problems, n_methods), dtype=numpy.float64)

    print('running HPHMM on synthetic problems')
    method_j = 0
    for problem_i in range(n_problems):
        p = problem[problem_i]
        hphmm = HybridPoissonHMM(
            transition_matrix=p.model.transition_matrix,
            signal_matrix=p.model.signal_matrix,
        )
        q0 = numpy.zeros((p.model.n, 3), dtype=numpy.float64)
        q0[:, 0] = p.state_prior
        q0[:, 1] = p.noise_model.prior_alpha
        q0[:, 2] = p.noise_model.prior_beta
        q, log_z = hphmm.forward(p.observations, q0)
        print('log_z = %r' % (log_z, ))
        log_z_by_problem_method[problem_i, method_j] = log_z

    for r in fghmm_rs:
        method_j += 1
        print('running fixed-Gamma HMM (r=%d) on synthetic problems' % (r, ))
        for problem_i in range(n_problems):
            p = problem[problem_i]

            base_model = p.model
            base_n = p.model.n

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
            prior_alpha = p.noise_model.prior_alpha
            prior_beta = p.noise_model.prior_beta
            midpoints = (1.0/r) * (0.5 + numpy.arange(r, dtype=numpy.float64))
            lambdas = scipy.stats.gamma.ppf(midpoints, a=prior_alpha, scale=(1.0 / prior_beta))
            fixed_betas = (prior_beta + p.n_timesteps) * numpy.ones((r, ), dtype=numpy.float64)
            fixed_alphas = lambdas * fixed_betas

            alpha_beta = numpy.zeros((r, 2), dtype=numpy.float64)
            alpha_beta[:, 0] = fixed_alphas
            alpha_beta[:, 1] = fixed_betas

            p0 = numpy.zeros((r, base_n), dtype=numpy.float64)
            for i in range(r):
                p0[i, :] = p.state_prior
            z = numpy.sum(p0)
            inv_z = 1.0 / z
            p0 *= inv_z

            fghmm = FixedGammaHMM(
                transition_matrix=base_model.transition_matrix,
                signal_matrix=base_model.signal_matrix,
                alpha_beta=alpha_beta,
            )

            p, log_z = fghmm.forward(p.observations, p0)
            print('log_z = %r' % (log_z,))
            log_z_by_problem_method[problem_i, method_j] = log_z
        print('done')

    print('results:')

    colheaders = '\t'.join('%8s' % (name, ) for name in method_names)
    print(colheaders)
    for i in range(n_problems):
        line = '\t'.join('%8.1f' % (log_z_by_problem_method[i, j], ) for j in range(n_methods))
        print(line)


if __name__ == '__main__':
    main()