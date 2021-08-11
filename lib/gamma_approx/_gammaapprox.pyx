# cython: profile=False

cimport cython
from cython cimport view
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

cdef struct BatchFitResult:
    int status
    size_t iters


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef BatchFitResult fit_batch_gamma_dists_to_gamma_mixtures(
        const index_t[:] lengths,
        const dtype_t[:] c,
        const dtype_t[:] a,
        const dtype_t[:] b,
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:
    cdef index_t n_mixtures, i, n, start, end
    cdef BatchFitResult result
    result.status = StatusInvalidInput
    result.iters = 0

    # Sanity check length packing
    n_mixtures = lengths.shape[0]
    n = 0
    for i in range(n_mixtures):
        n += lengths[i]
    if n != c.shape[0]:
        return result
    if n != a.shape[0]:
        return result
    if n != b.shape[0]:
        return result
    if n_mixtures != out_alpha.shape[0]:
        return result
    if n_mixtures != out_beta.shape[0]:
        return result

    start = 0
    end = 0
    for i in range(n_mixtures):
        end += lengths[i]
        fit_result = fit_gamma_dist_to_gamma_mixture(
            c[start:end],
            a[start:end],
            b[start:end],
        )
        result.iters += fit_result.iters
        if fit_result.error:
            return result
        out_alpha[i] = fit_result.alpha
        out_beta[i] = fit_result.beta
        start += lengths[i]
    result.status = StatusOK
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

    p = sum_i c_i \frac{a_i}{b_i} is the input mixture to approximate, and

    theta = (alpha, beta), alpha>0, beta>0 are the parameters of the
    approximating Gamma distribution q.

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

    # Naive guess of alpha^* as convex combination of mixture alphas.
    x0 = fancyish_initial_guess(y, c, a)

    hresult = inverse_digamma_minus_log_halley(y, x0=x0)
    a_star = hresult.x
    b_star = a_star / expected_lambda
    result.alpha = a_star
    result.beta = b_star
    result.error = hresult.error
    result.iters = hresult.iters
    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dtype_t fancyish_initial_guess(const dtype_t y, const dtype_t[:] c, const dtype_t[:] a) nogil:
    cdef dtype_t x0
    cdef index_t n
    n = c.shape[0]
    if y >= 0.0:
        # Naive guess of alpha^* as convex combination of mixture alphas.
        x0 = 0.0
        for i in range(n):
            x0 += c[i] * a[i]
    else:
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
        x0 = 0.6548254483045827 * pow(-y, -0.9256158)
    return x0


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
        acc += c[i] * (_approx_digamma(a[i]) - log(b[i]))
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

    x_1 := x_0 - f(x_0)/f'(x_0) [1 - (f(x_0)/f'(x_0))*(f''(x_0)/(2f'(x_0)))]^-1

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
    """
    cdef dtype_t digamma_0 = 0.0, digamma_1 = 0.0, digamma_2 = 0.0
    cdef dtype_t f_0, f_1, f_2, ff_0
    cdef dtype_t x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
    cdef HalleyStep result
    x = x0

    # The series approximation of digamma(x) is only accurate for sufficiently
    # large x. We use the recurrence relation digamma(x) = -x^-1 + digamma(x+1)
    # to reduce the problem for small x to a problem for large x.
    # Differentiating the recurrence produces similar recurrences for the first
    # and second derivatives.
    while x < 7:
        digamma_0 -= 1.0 / x
        digamma_1 += 1.0 / (x * x)
        digamma_2 -= 2.0 / (x * x * x)
        x += 1
    x -= 1.0 / 2.0
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
    digamma_0 += log(x) +   (1./24.) * x_2 -   (7.0/960.0) * x_4 +   (31.0/8064.0) * x_6  - (127.0/30720.0) * x_8
    digamma_1 +=    x_1 - (2.0/24.0) * x_3 +  (28.0/960.0) * x_5 -  (186.0/8064.0) * x_7 + (1016.0/30720.0) * x_9
    digamma_2 +=   -x_2 + (6.0/24.0) * x_4 - (140.0/960.0) * x_6 + (1302.0/8064.0) * x_8 - (9144.0/30720.0) * x_10

    f_0 = digamma_0 - log(x0) - y
    f_1 = digamma_1 - 1.0 / x0
    f_2 = digamma_2 + 1.0 / (x0 * x0)
    ff_0 = f_0 / f_1
    result.f = f_0
    result.step = -ff_0 / (1 - ff_0 * f_2 / (2.0 * f_1))
    return result



# Adapted from Mark Johnson's digamma.c
# ref: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
@cython.cdivision(True)
cdef inline dtype_t _approx_digamma(dtype_t x) nogil except 0.0:
    cdef dtype_t result = 0, xx, xx2, xx4
    while x < 7:
        result -= 1.0/x
        x += 1
    x -= 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
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