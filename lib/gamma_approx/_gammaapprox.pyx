# cython: profile=False

cimport cython
from libc.math cimport pow, log, fabs, NAN

ctypedef double dtype_t
ctypedef int int_t
ctypedef Py_ssize_t index_t


cdef struct HalleyStep:
    dtype_t f
    dtype_t step


cdef struct HalleyResult:
    dtype_t x
    size_t iters
    bint error

cdef struct FitResult:
    dtype_t alpha
    dtype_t beta
    bint error
    size_t iters

DEF StatusOK = 0
DEF StatusInvalidInput = 1
DEF FailedToConverge = 2

cdef struct BatchFitResult:
    int status
    size_t iters
    size_t n_issues


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef BatchFitResult validate_batch_input(
        const index_t[:] lengths,
        const dtype_t[:,::1] cab,
        const dtype_t[:] out_alpha,
        const dtype_t[:] out_beta) nogil:

    cdef index_t n_mixtures, i, j, n, m, start
    cdef dtype_t acc_c
    cdef BatchFitResult result

    result.status = StatusInvalidInput
    result.iters = 0
    result.n_issues = 0

    # Sanity check input shapes conform with length packing.
    n_mixtures = lengths.shape[0]
    n = 0
    for i in range(n_mixtures):
        n += lengths[i]
    result.n_issues += (n != cab.shape[0])
    result.n_issues += (3 != cab.shape[1])
    result.n_issues += (n_mixtures != out_alpha.shape[0])
    result.n_issues += (n_mixtures != out_beta.shape[0])

    if result.n_issues > 0:
        return result

    # Sanity check data is in domain
    for i in range(n):
        result.n_issues += (cab[i, 0] < 0.0)
        result.n_issues += (cab[i, 1] <= 0.0)
        result.n_issues += (cab[i, 2] <= 0.0)

    # Sanity check mixture coefficients are convex combinations
    start = 0
    for i in range(n_mixtures):
        m = lengths[i]
        acc_c = 0.0
        for j in range(m):
            acc_c += cab[start + j, 0]
        start += m
        result.n_issues += fabs(acc_c - 1.0) > 1e-8

    if result.n_issues > 0:
        return result

    result.status = StatusOK
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef BatchFitResult fit_batch_gamma_dists_to_gamma_mixtures(
        const index_t[:] lengths,
        const dtype_t[:,::1] cab, # shape (n_components, 3)
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:
    cdef index_t n_mixtures, i, j, n, m, start
    cdef dtype_t expected_lambda, expected_log_lambda, acc_c, cj, aj, bj, ratej, y
    cdef BatchFitResult result
    cdef HalleyResult hresult

    result = validate_batch_input(lengths, cab, out_alpha, out_beta)
    if result.status != StatusOK:
        return result

    n_mixtures = lengths.shape[0]
    start = 0
    for i in range(n_mixtures):
        m = lengths[i]
        expected_lambda = 0.0
        expected_log_lambda = 0.0
        for j in range(m):
            cj = cab[start+j, 0]
            aj = cab[start+j, 1]
            bj = cab[start+j, 2]
            ratej = aj / bj
            expected_lambda += cj * ratej
            expected_log_lambda += cj * _approx_digamma_minus_log_b(aj, bj)
        y = expected_log_lambda - log(expected_lambda)
        hresult = inverse_digamma_minus_log_halley(y, x0=rough_inverse_f(y))
        out_alpha[i] = hresult.x
        out_beta[i] = hresult.x / expected_lambda
        result.status |= (FailedToConverge * hresult.error)
        result.n_issues += hresult.error
        result.iters += hresult.iters
        start += m
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef BatchFitResult rough_fit_batch_gamma_dists_to_gamma_mixtures(
        const index_t[:] lengths,
        const dtype_t[:,::1] cab, # shape (n_components, 3)
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:
    """
    rough_fit_batch_gamma_dists_to_gamma_mixtures is a variant of
    fit_batch_gamma_dists_to_gamma_mixtures which performs exactly two
    Halley-method iterations when approximating the inverse of
    f(x) = digamma(x) - log(x).

    Normally, we perform a variable number of Halley-method iterations based on
    an error estimate, with a mean of slightly fewer than two iterations.

    Hardcoding two iterations and removing all logic controlling iterations and
    convgergence checks reduces the running time by around 11%.
    """
    cdef BatchFitResult result

    result = validate_batch_input(lengths, cab, out_alpha, out_beta)
    if result.status != StatusOK:
        return result

    # Breaking the computation into two passes doesn't give any speedup, but
    # it does make it easier to understand the performance.

    # Compute the expected rate and expected log rate of each mixture in batch.
    # out_alpha and out_beta are (ab)used as temporary storage of results.
    # note: this pass takes 58% running time.
    batch_expected_rate_expected_log_rate(lengths, cab, out_alpha, out_beta)

    # Compute the single Gamma fit from expected rate & expected log rate
    # note: this pass takes 42% running time.
    result = batch_fit_from_expectations(lengths, cab, out_alpha, out_beta)

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void batch_expected_rate_expected_log_rate(
        const index_t[:] lengths,
        const dtype_t[:,::1] cab,
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:

    cdef index_t n_mixtures, i, j, m, start
    cdef dtype_t expected_lambda, expected_log_lambda, cj, aj, bj, ratej

    # Rough compute time breakdown:
    # 40% is the single _log call made by approx_digamma_minus_log_b
    # 50% is the rest of _approx_digamma_minus_log_b
    # 10% everything else

    n_mixtures = lengths.shape[0]
    start = 0
    for i in range(n_mixtures):
        m = lengths[i]
        expected_lambda = 0.0
        expected_log_lambda = 0.0
        for j in range(m):
            cj = cab[start + j, 0]
            aj = cab[start + j, 1]
            bj = cab[start + j, 2]
            ratej = aj / bj
            expected_lambda += cj * ratej
            expected_log_lambda += cj * _approx_digamma_minus_log_b(aj, bj)
        out_alpha[i] = expected_lambda
        out_beta[i] = expected_log_lambda
        start += m
    return


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef BatchFitResult batch_fit_from_expectations(
        const index_t[:] lengths,
        const dtype_t[:,::1] cab,
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:

    cdef index_t n_mixtures, i
    cdef dtype_t y, x, expected_lambda, expected_log_lambda
    cdef HalleyStep hs

    n_mixtures = lengths.shape[0]

    cdef BatchFitResult result
    result.status = StatusOK
    result.n_issues = 0
    result.iters = 2 * n_mixtures

    # Rough compute time breakdown:
    # 37% for first halley step
    # 37% for second halley step
    # 26% for everything else

    for i in range(n_mixtures):
        expected_lambda = out_alpha[i]
        expected_log_lambda = out_beta[i]
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
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FitResult fit_gamma_dist_to_gamma_mixture(
        const dtype_t[:] c,
        const dtype_t[:] a,
        const dtype_t[:] b) nogil:
    r"""
    Approximate the given mixture of n Gamma distributions with a single
    Gamma distribution.

    The returned fit minimises the Kullback–Leibler divergence (also
    known as the relative entropy) of the mixture from the approximation:

    theta^* = argmin_theta D_KL(p(\lambda) || q(\lambda | theta))

    where

        p(\lambda) = sum_i c_i Gamma(\lambda ; alpha=a_i, beta=b_i)

    is the input mixture to approximate, and

    theta = (alpha, beta), alpha>0, beta>0 are the parameters of the
    approximating Gamma distribution

        q(\lambda | theta) = Gamma(\lambda | alpha, beta)

    Minimising the KL divergence reduces to finding a Gamma distribution q that
    satisfies the constraints:

        E_p [ \lambda ]         =   E_q [ \lambda ]
        E_p [ log(\lambda) ]    =   E_q [ log(\lambda) ]

    This system of constraints may be obtained analytically by computing the
    gradient of D_KL(p(\lambda) || q(\lambda | theta)) with respect to theta
    and setting it equal to the zero vector.

    :param c: shape (n, ) array of coefficients giving convex combination.
    :param a: shape (n, ) array of Gamma distribution shape parameters alpha > 0
    :param b: shape (n, ) array of Gamma distribution rate parameters beta > 0
    :return: shape (2, ) array (alpha_star, beta_star) of best fit.
    """
    cdef index_t n, i
    cdef dtype_t acc_c, expected_lambda, expected_log_lambda, y, x0, a_star, b_star
    cdef FitResult result
    cdef HalleyResult hresult

    # Validate input
    result.alpha = NAN
    result.beta = NAN
    result.error = True
    result.iters = 0

    n = c.shape[0]
    if n != a.shape[0] or n != b.shape[0] or n < 1:
        return result

    acc_c = 0.0
    for i in range(n):
        if a[i] <= 0.0 or b[i] <= 0.0 or c[i] < 0.0:
            return result
        acc_c += c[i]
    if fabs(acc_c - 1.0) > 1e-8:
        return result

    # Special case trivial case of 1 component distribution in mixture
    if n == 1:
        result.alpha = a[0]
        result.beta = b[0]
        result.error = False
        return result

    expected_lambda = expected_rate_of_gamma_mixture(c, a, b)
    expected_log_lambda = expected_log_rate_of_gamma_mixture(c, a, b)
    y = expected_log_lambda - log(expected_lambda)

    hresult = inverse_digamma_minus_log_halley(y, x0=rough_inverse_f(y))
    a_star = hresult.x
    b_star = a_star / expected_lambda
    result.alpha = a_star
    result.beta = b_star
    result.error = hresult.error
    result.iters = hresult.iters
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
cdef inline dtype_t rough_f(const dtype_t x) nogil:
    # Approximate f(x) = digamma(x) - log(x)
    # assumption: x > 0
    return -1.0 * pow(1.5271245224038152 * x , -1.0803618520772873)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dtype_t expected_rate_of_gamma_mixture(
        const dtype_t[:] c,
        const dtype_t[:] a,
        const dtype_t[:] b) nogil:
    r"""
    Computes the expected value of the rate parameter lambda of the given
    mixture of n Gamma distributions,

        E_p[\lambda]

    where p := sum_i c_i \frac{a_i}{b_i}

    :param c: shape (n, ) array of coefficients giving convex combination.
    :param a: shape (n, ) array of Gamma distribution shape parameters alpha > 0
    :param b: shape (n, ) array of Gamma distribution rate parameters beta > 0
    :return: scalar floating point value - expected rate parameter
    """
    cdef dtype_t acc = 0.0
    cdef index_t i, n
    n = c.shape[0]
    for i in range(n):
        acc += c[i] * (a[i] / b[i])
    return acc


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dtype_t expected_log_rate_of_gamma_mixture(
        const dtype_t[:] c,
        const dtype_t[:] a,
        const dtype_t[:] b) nogil:
    r"""
    Computes the expected value of the natural log of the rate parameter
    lambda of the given mixture of n Gamma distributions,

        E_p[log(\lambda)]

    where p := sum_i c_i \frac{a_i}{b_i}

    This is evaluated using the identity

    E_g[(log(\lambda)] = psi(a) - log(b)

    where g=Gamma(\lambda ; a, b) is a Gamma function and psi is the digamma
    function.

    :param c: shape (n, ) array of coefficients giving convex combination.
    :param a: shape (n, ) array of Gamma distribution shape parameters alpha > 0
    :param b: shape (n, ) array of Gamma distribution rate parameters beta > 0
    :return: scalar floating point value - expected log(rate) parameter
    """
    cdef dtype_t acc = 0.0
    cdef index_t i, n
    n = c.shape[0]
    for i in range(n):
        acc += c[i] * _approx_digamma_minus_log_b(a[i], b[i])
    return acc


@cython.cdivision(True)
cdef inline HalleyResult inverse_digamma_minus_log_halley(
        dtype_t y, \
        dtype_t x0, \
        index_t max_iters=50, \
        dtype_t rtol=1.0e-14, \
        dtype_t atol=1.0e-4, \
    ) nogil:
    """
    Computes x > 0 such that y = psi(x) - log(x) from an initial guess x0 > 0.

    Uses Halley's method [1] to iteratively solve for y. Halley's method uses
    the iteration:

    x_1 := x_0 - 2 f(x_0) f'(x_0) / (2 f'(x_0)^2 - f(x_0) f''(x_0))

    where

          f(x) = psi(x) - log(x) - y
         f'(x) = psi'(x) - 1/x          = polygamma[1](x) - 1/x
        f''(x) = psi''(x) + 1/x^2       = polygamma[2](x) + 1/x^2

    This iterative scheme is inspired by a technical report by Minka [2]
    which explains how to compute the inverse of the digamma function psi(x)
    using Newton's method, albeit with the following differences:

    -   The function to invert, f(x), has an additional -log(x) term in
        addition to the digamma(x) term.
    -   We use Halley's method instead of Newton's method. f(x) has a
        singularity at x=0, and as x approaches zero from above, f(x) becomes
        highly concave. This can cause Newton's method to take a step to a
        negative value of x on the other side of the singularity, which leads
        to divergence. Halley's method is able to exploit more information
        about the curvature of f(x) and does not appear to be afflicted by this
        problem.
    -   When evaluating f(x), the digamma(x) term is approximated by the
        series expansion and C code from Mark Johnson's digamma.c [3]. The
        first and second derivatives of digamma(x) are approximated by the
        first and second derivatives of the series expansion respectively.
    -   Minka calculates a cheap to compute initial guess for x0 based on
        inverting some asymptotic approximations of digamma(x). We don't use
        an analytically obtained asymptotic approximation but compute the
        graph of the inverse numerically over a range of interest, then fit
        a curve to it, and use that approximating curve as an initial guess.

    [1] https://en.wikipedia.org/wiki/Halley%27s_method

    [2] https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    @misc{minka2000estimating,
      title={Estimating a Dirichlet distribution},
      author={Minka, Thomas},
      year={2000},
      publisher={Technical report, MIT}
    }
    See Appendix C "Inverting the Psi function".

    [3] http://web.science.mq.edu.au/~mjohnson/code/digamma.c
    """

    cdef index_t i
    cdef dtype_t fprev, f, x1, xgap, fgap
    cdef bint cvgc
    cdef HalleyStep hs
    cdef HalleyResult result

    i = 0

    fprev=NAN # not possible

    while True:
        hs = _approx_f_halley_step(x0, y)
        f = hs.f

        # Adhoc guard against stepping to the other side of the singularity at
        # x=0. Unclear if this guard is necessary for Halley's method. The
        # guard is necessary if using Newton-Raphson method.
        if x0 + hs.step <= 0.0:
            x1 = 0.25 * x0
        else:
            x1 = x0 + hs.step

        xgap = fabs(x1 - x0)
        fgap = fabs(f - fprev)
        cvgc = fgap <= atol
        cvgc |= fgap <= rtol * max(fabs(f), fabs(fprev))
        cvgc |= xgap <= rtol * max(fabs(x0), fabs(x1))
        if cvgc:
            result.x = x1
            result.error = False
            result.iters = i
            return result
        if i >= max_iters:
            break
        fprev = f
        x0 = x1
        i += 1

    result.x = x1
    result.error = True
    result.iters = i
    return result


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


# Approximation of derivative of digamma(x) aka polygamma(1).
# This is merely the termwise derivative of the approximation above.
# See also: scipy.special.polygamma(1, x)
@cython.cdivision(True)
cdef inline dtype_t _approx_digamma_prime(dtype_t x) except 0.0:
    cdef dtype_t result = 0, x_1, x_2, x_3, x_5, x_7, x_9
    assert x > 0
    while x < 7:
        result += 1.0/(x*x)
        x += 1
    x -= 1.0/2.0
    x_1 = 1.0/x
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_5 = x_3 * x_2
    x_7 = x_5 * x_2
    x_9 = x_7 * x_2
    result += x_1 - (2.0/24.0) * x_3 + (28.0/960.0) * x_5 - (186.0/8064.0) * x_7 + (1016.0/30720.0) * x_9
    return result


# Approximation of 2nd derivative of digamma(x) aka polygamma(2).
# This is merely the termwise derivative of the approximation above.
# See also: scipy.special.polygamma(1, x)
@cython.cdivision(True)
cdef inline dtype_t _approx_digamma_prime2(dtype_t x) except 0.0:
    cdef dtype_t result = 0, x_1, x_2, x_4, x_6, x_8, x_10
    assert x > 0
    while x < 7:
        result -= 2.0/(x*x*x)
        x += 1
    x -= 1.0/2.0
    x_1 = 1.0/x
    x_2 = x_1 * x_1
    x_4 = x_2 * x_2
    x_6 = x_4 * x_2
    x_8 = x_4 * x_4
    x_10 = x_8 * x_2
    result += -x_2 + (6.0/24.0) * x_4 - (140.0/960.0) * x_6 + (1302.0/8064.0) * x_8 - (9144.0/30720.0) * x_10
    return result