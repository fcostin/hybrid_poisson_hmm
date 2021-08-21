cimport cython
from libc.math cimport tgamma as gamma, pow, log

ctypedef double dtype_t
ctypedef int int_t
ctypedef Py_ssize_t index_t

DEF StatusOK = 0
DEF StatusInvalidInput = 1
DEF FailedToConverge = 2

cdef struct HalleyStep:
    dtype_t f
    dtype_t step

cdef struct BatchFitResult:
    int status
    size_t iters
    size_t n_issues


import numpy


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef const dtype_t[:, :] forward(
        const dtype_t[:, :] transition_matrix,
        const dtype_t[:, :] signal_matrix,
        const int_t[:] observations,
        const dtype_t[:, :] q0):

    cdef index_t n, max_k, max_p, y_t, k, w_lo, w_hi, p, np, w, i, j, t, n_obs, start

    cdef dtype_t alpha_, beta_, neg_bin_w_a_b, z_i, z

    cdef BatchFitResult result

    n = q0.shape[0]
    n_obs = observations.shape[0]
    max_k = signal_matrix.shape[0] - 1

    cdef const dtype_t[:, :] q = q0

    cdef dtype_t[:, :] q_prime = numpy.zeros(shape=(n, 3), dtype=numpy.float64)  # work buffer.

    cdef index_t[:] lengths = numpy.zeros((1,), dtype=numpy.int64)  # FIXME alloc - bad. change API

    max_p = 20 # implementation-defined limitation
    cdef dtype_t[:, :] common_cab = numpy.zeros((n * max_p, 3), dtype=numpy.float64)  # work buffer.

    cdef dtype_t[:, ::1] otq_i_cab = numpy.empty((n * max_p, 3), dtype=numpy.float64) # work buffer.

    cdef dtype_t[:] c = numpy.empty((n, ), dtype=numpy.float64) # work buffer.

    cdef dtype_t[:, :] chi = numpy.zeros((n * max_p, 2), dtype=numpy.float64)  # work buffer.

    cdef dtype_t[:] expected_lambdas = numpy.zeros(shape=(n, ), dtype=numpy.float64)  # work buffer.
    cdef dtype_t[:] expected_log_lambdas = numpy.zeros(shape=(n, ), dtype=numpy.float64)  # work buffer.

    for t in range(n_obs):
        k = observations[t]

        # Discrete convolution of signal with noise wrt observation k.
        # Signal matrix[k-w] is zero unless 0 <= k-w <= max_k
        # Neg-Bin(w, alpha, beta) is zero unless w >= 0
        # equivalently:
        # 0 <= w
        # - max_k+ k <= w
        # w <= k
        w_lo = max(0, -max_k + k)
        w_hi = k + 1
        p = w_hi - w_lo

        assert p <= max_p # artificial limit to bound work buffer size

        np = n*p

        lengths[0] = np

        common_cab[:np, :] = 0.0
        for w in range(w_lo, w_hi):
            j = w - w_lo
            start = (n * j)
            for i in range(n):
                alpha_ = q[i, 1] + w
                beta_ = q[i, 2] + 1.0

                # FIXME computing neg_bin_w_a_b costs approx 15% running time.
                neg_bin_w_a_b = (
                        (gamma(alpha_ + w) / (gamma(w + 1) * gamma(alpha_))) *
                        ((beta_ / (beta_ + 1)) ** alpha_) *
                        ((1.0 / (beta_ + 1)) ** w)
                )

                common_cab[start + i, 0] = signal_matrix[k - w][i] * neg_bin_w_a_b
                common_cab[start + i, 1] = alpha_
                common_cab[start + i, 2] = beta_

        # common_cab forms weighted "basis" of Gammas

        # Precompute E[rate] and E[log(rate)] for each element of our Gamma
        # "basis". digamma and logs are expensive to compute. This avoids
        # recomputing digamma and log for terms that are shared in many
        # mixture-sums.
        # FIXME in previous block we see many alpha differ by integer amounts
        #  -- could try to exploit digamma(z+1) = digamma(z) + 1/z
        for i in range(np):
            alpha_ = common_cab[i, 1]
            beta_ = common_cab[i, 2]
            chi[i, 0] = alpha_ / beta_
            chi[i, 1] = _approx_digamma_minus_log_b(alpha_, beta_)
        # TODO we dont need E[log(rate)], we really need E[log(rate)] - log(E[rate]).

        # i : destination state index
        for i in range(n):
            otq_i_cab[:np, 0] = common_cab[:np, 0]  # FIXME eliminate copy ?
            otq_i_cab[:np, 1] = chi[:np, 0]  # FIXME eliminate copy ?
            otq_i_cab[:np, 2] = chi[:np, 1]  # FIXME eliminate copy ?

            # FIXME dense matrix. reimplement as sparse.
            for j in range(n):
                c[j] = transition_matrix[i, j] * q[j, 0]
            for w in range(w_lo, w_hi):
                start = (n * (w - w_lo))
                for j in range(n):
                    otq_i_cab[start+j, 0] *= c[j]

            z_i = 0.0
            for j in range(np):
                z_i += otq_i_cab[j, 0]
            for j in range(np):
                otq_i_cab[j, 0] = otq_i_cab[j, 0] / z_i

            q_prime[i, 0] = z_i

            # Compute expected lambda and expected log lambda for mixture
            # of Gamma distributions associated with destination state index i
            expected_lambdas[i] = 0.0
            expected_log_lambdas[i] = 0.0
            for j in range(np):
                expected_lambdas[i] += otq_i_cab[j, 0] * otq_i_cab[j, 1]
                expected_log_lambdas[i] += otq_i_cab[j, 0] * otq_i_cab[j, 2]

        result = batch_fit_from_expectations(
            expected_lambdas,
            expected_log_lambdas,
            q_prime[:, 1],
            q_prime[:, 2],
        )
        assert result.status == 0, repr(result)

        # Normalise
        z = 0.0
        for i in range(n):
            z += q_prime[i, 0]
        assert z > 0.0
        for i in range(n):
            q_prime[i, 0] = q_prime[i, 0] / z
        q = q_prime
    return q


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef BatchFitResult batch_fit_from_expectations(
        const dtype_t[:] expected_lambdas,
        const dtype_t[:] expected_log_lambdas,
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:

    cdef index_t n_mixtures, i
    cdef dtype_t y, x, expected_lambda, expected_log_lambda
    cdef HalleyStep hs

    n_mixtures = expected_lambdas.shape[0]

    cdef BatchFitResult result
    result.status = StatusOK
    result.n_issues = 0
    result.iters = 2 * n_mixtures

    # Rough compute time breakdown:
    # 37% for first halley step
    # 37% for second halley step
    # 26% for everything else

    for i in range(n_mixtures):
        expected_lambda = expected_lambdas[i]
        expected_log_lambda = expected_log_lambdas[i]
        y = expected_log_lambda - log(expected_lambda)
        x = rough_inverse_f(y)
        hs = _approx_f_halley_step(x, y)
        x += hs.step
        hs = _approx_f_halley_step(x, y)
        x += hs.step
        out_alpha[i] = x
        out_beta[i] = x / expected_lambda
    return result


@cython.cdivision(True)
cdef inline dtype_t rough_inverse_f(const dtype_t y) nogil:
    # Approximate inverse f^-1(y) where f(x) = digamma(x) - log(x)
    # assumption: y < 0
    #
    # Explanation of this approximation:
    #
    # Let x_i range over the grid numpy.geomspace(0.001, 100.0, 3000)
    # Define y_i := f(x_i) for f(x) = digamma(x) - log(x)
    #
    # Note that we used scipy's digamma implementation to define the
    # data y_i, unrelated to the digamma approximations in this file.
    #
    # Assume the relationship between x and y can be approximated for
    # x in [0.001, 100] by:
    #
    # log(x) = a + b log(-1/y)
    #
    # This isn't motivated by theory, but empirically, by graphing
    # log(x) as a function of y and making a guess.
    #
    # Fit parameters (a, b) using least-squares regression evaluated
    # over the points y_i to approximate the target function log(x_i).
    #
    # Then x is approximated by exp(a + b log(-1/y)) = exp(a) (-y)^b
    #
    # The parameters (a, b) found from the procedure above are:
    #
    # a = -0.42338657
    # b =  0.9256158
    #
    # We can rearrange for x0 to give:  x0 approx 0.6548 * pow(-y, -0.9256)
    #
    # This gives us a relatively cheap way to initialise a guess before
    # starting Halley's method iterations when attempting to invert f.
    #
    # After switching to this initial guess for negative y, instead of
    # the naive guess used above, it reduced the mean number of Halley's
    # method iterations required to invert f on a testbed of 80,000
    # small mixture approximation problems from 3.07 to 2.00.
    return 0.6548254483045827 * pow(-y, -0.9256158)


@cython.cdivision(True)
cdef inline HalleyStep _approx_f_halley_step(dtype_t x0, dtype_t y) nogil:
    """
    Computes a step of Halley method [1] for an approximation of the function

    f(x) = f(x;y) at point x=x_0 > 0.

    where f(x) := psi(x) - log(x) - y

    x1 = x0 + step
    where
    step = - 2 f0 f1 / (2 f1^2 - f0 f2)
    """
    cdef dtype_t digamma_0_minus_log_x0 = 0.0, digamma_1 = 0.0, digamma_2 = 0.0
    cdef dtype_t f_0, f_1, f_2
    cdef dtype_t x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
    cdef dtype_t z0, z1, z2, z3, z4, z5, z6
    cdef HalleyStep result


    # The series approximation of digamma(x) is only accurate for sufficiently
    # large x. We use the recurrence relation digamma(x) = -x^-1 + digamma(x+1)
    # to reduce the problem for small x to a problem for large x.
    # Differentiating the recurrence produces similar recurrences for the first
    # and second derivatives.

    # Minor optimisation (around -0.7% running time of a rough batch fit): this
    # used to be a "while x < 7:" loop. x must be positive, so the loop runs at
    # most 7 times. Most values of x we see are small. If we run the loop
    # exactly 7 times regardless of the value of x then we completely eliminate
    # branches at the cost of occasionally doing some unnecessary work.

    z0 = 1.0/x0
    z1 = 1.0/(x0+1.0)
    z2 = 1.0/(x0+2.0)
    z3 = 1.0/(x0+3.0)
    z4 = 1.0/(x0+4.0)
    z5 = 1.0/(x0+5.0)
    z6 = 1.0/(x0+6.0)

    digamma_0_minus_log_x0 = -z0 - z1 -z2 -z3 -z4 -z5 -z6
    digamma_1 = z0*z0 + z1*z1 + z2*z2 + z3*z3 + z4*z4 + z5*z5 + z6*z6
    digamma_2 = -2.0 * (z0*z0*z0 + z1*z1*z1 + z2*z2*z2 + z3*z3*z3 + z4*z4*z4 + z5*z5*z5 + z6*z6*z6)

    x = x0 + 7.0 - 0.5
    x_1 = 1.0 / x
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_4 = x_2 * x_2
    x_5 = x_3 * x_2
    x_6 = x_4 * x_2
    x_7 = x_5 * x_2
    x_8 = x_4 * x_4
    x_9 = x_7 * x_2
    x_10 = x_8 * x_2
    # We do log(x/x0) in preference to log(x) - log(x0) to reduce log calls.
    # where log(x) is part of digamma0 but -log(x0) is not.
    digamma_0_minus_log_x0 += log(x/x0) +   (1./24.) * x_2 -   (7.0/960.0) * x_4 +   (31.0/8064.0) * x_6  - (127.0/30720.0) * x_8
    digamma_1 +=    x_1 - (2.0/24.0) * x_3 +  (28.0/960.0) * x_5 -  (186.0/8064.0) * x_7 + (1016.0/30720.0) * x_9
    digamma_2 +=   -x_2 + (6.0/24.0) * x_4 - (140.0/960.0) * x_6 + (1302.0/8064.0) * x_8 - (9144.0/30720.0) * x_10

    # combining the two log calls into one speeds up pass1 by about 4%

    f_0 = digamma_0_minus_log_x0 - y
    f_1 = digamma_1 - z0
    f_2 = digamma_2 + (z0*z0)
    result.f = f_0
    result.step = -2.0 * f_0 * f_1 / (2.0 *  f_1 * f_1 - f_0 * f_2)
    return result


@cython.cdivision(True)
cdef inline dtype_t _approx_digamma_minus_log_b(const dtype_t x0, const dtype_t b) nogil except 0.0:
    cdef dtype_t result = 0, x, xx, xx2, xx4
    result = -1.0/x0 - 1.0/(x0+1.0) -1.0/(x0+2.0) -1.0/(x0+3.0) -1.0/(x0+4.0) -1.0/(x0+5.0) -1.0/(x0+6.0)

    x = x0 + 7.0 - 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x/b)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
    return result


# Adapted from Mark Johnson's digamma.c
# ref: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
@cython.cdivision(True)
cdef inline dtype_t _approx_digamma(const dtype_t x0) nogil except 0.0:
    cdef dtype_t result = 0, x, xx, xx2, xx4

    # This used to be a "while x < 7" loop. If most x values are small, then we need
    # to do 7 iterations anyway. This form has no branches, it is also one big
    # expression that doesn't suggest a particular eval order. Let the compiler
    # and the hardware decide.
    result = -1.0/x0 - 1.0/(x0+1.0) -1.0/(x0+2.0) -1.0/(x0+3.0) -1.0/(x0+4.0) -1.0/(x0+5.0) -1.0/(x0+6.0)

    x = x0 + 7.0 - 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
    return result