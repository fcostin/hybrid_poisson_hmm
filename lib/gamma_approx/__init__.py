import numpy
from numpy.typing import NDArray
from scipy.special import digamma

from ._gammaapprox import (
    fit_batch_gamma_dists_to_gamma_mixtures,
    fit_gamma_dist_to_gamma_mixture,
    rough_fit_batch_gamma_dists_to_gamma_mixtures,
)


def expected_rate_of_gamma_mixture(
        c: NDArray,
        a: NDArray,
        b: NDArray) -> float:
    r"""
    Computes the expected value of the rate parameter lambda of the given
    mixture of n Gamma distributions,

        E_p[\lambda]

    where p := sum_i c_i \frac{a_i}{b_i}

    note: this is a simple, slow reference implementation.

    :param c: shape (n, ) array of coefficients giving convex combination.
    :param a: shape (n, ) array of Gamma distribution shape parameters alpha > 0
    :param b: shape (n, ) array of Gamma distribution rate parameters beta > 0
    :return: scalar floating point value - expected rate parameter
    """
    return numpy.sum(c * (a / b), axis=0)


def expected_log_rate_of_gamma_mixture(
        c: NDArray,
        a: NDArray,
        b: NDArray) -> float:
    r"""
    Computes the expected value of the natural log of the rate parameter
    lambda of the given mixture of n Gamma distributions,

        E_p[log(\lambda)]

    where p := sum_i c_i \frac{a_i}{b_i}

    This is evaluated using the identity

    E_g[(log(\lambda)] = psi(a) - log(b)

    where g=Gamma(\lambda ; a, b) is a Gamma function and psi is the digamma
    function.

    note: this is a simple, slow reference implementation.

    :param c: shape (n, ) array of coefficients giving convex combination.
    :param a: shape (n, ) array of Gamma distribution shape parameters alpha > 0
    :param b: shape (n, ) array of Gamma distribution rate parameters beta > 0
    :return: scalar floating point value - expected log(rate) parameter
    """

    return numpy.sum(c * (digamma(a) - numpy.log(b)), axis=0)

