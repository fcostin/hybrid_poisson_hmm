# hphmm - hybrid hidden Markov models with Poisson noise

## Overview

hphmm implements an approximate hybrid hidden Markov
model for event counts with Poisson noise. The posterior
distribution is approximated by a factorisation consisting
of an approximate conditional posterior distribution for
the Poisson noise rate given the hidden state, and a
discrete posterior distribution over the hidden state.

The approximation scheme is an instance of assumed density
filtering: a single Gamma distribution approximates each
conditional posterior distribution for the Poisson noise rate.

### License

BSD

### Status

Prototype. Library API may change.


HMM capabilities -- implemented:

*	"filtering" -- estimating the posterior distribution at the
	current time `t` over latent variables given evidence to date `y_{1:t}`

HMM capabilities -- not implemented:

*	"smoothing" -- estimating the posterior distribution over latent
	variables given the evidence to date, for some historical time `t < T`
*	"prediction" -- estimating the posterior distribution over latent
	variables at future times `t > T` given evidence to date
*	most likely explanation -- recovering a trajectory of latent states
	that is best, in some particular sense (e.g. via Viterbi algorithm)
*	estimation -- estimating the parameters of the Markov model from
	training data (e.g. via the Baum-Welch algorithm)

Lower-level approximation capabilities -- implemented:

*	approximation of a mixture of Gamma distribution by a single Gamma
	distribution that minimises the KL divergence from the approximation
	to the original mixture.


### Model

Consider a discrete hidden Markov model which emits an
integer event count `z_t` each time step, that is combined
with an additive Poisson noise `k_t` to produce an
observed event count `y_t = z_t + k_t`:

![hmm-poisson-noise.png](./gallery/hmm-poisson-noise.png)

The posterior distribution over the hidden state `x_t` and
the noise rate `\lambda` can be decomposed as

```
P(x_t, \lambda | y_{1:t}) = P(\lambda | x_t, y_{1:t} ) P( x_t | y_{1:T})
```

where the first factor can be approximated by a Gamma distribution

```
P(\lambda | x_t, y_{1:t} ) \approx Gamma(\lambda ; \alpha, \beta)
```

where the parameters `\alpha, \beta` of the Gamma distribution depend
upon the current time step `t` and the hidden state `x_t`.


## `hphmm` library

### Purpose:

`hphmm` is a Python library implementing approximate HMM algorithm
for a discrete HMM with Poisson noise.

At present, `hphmm` is a rough prototype.


## `gamma_approx` library

### Purpose:

`gamma_approx` is a Python library to approximate mixtures
of Gamma distributions with a single Gamma distribution.

![example-gamma-mixture-plot](./gallery/example-gamma-mixture-approx.png)

### Quickstart for local development

1.	clone the repo
2.	ensure you have a dev environment with a C compiler
	that Cython can use
3.	Install deps, build the library and run tests:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make clean
make lib
python -m pytest lib
```

### Usage

The extension library `gamma_approx/_gammaapprox` defines
the following functions:

*	`fit_gamma_dist_to_gamma_mixture`
*	`fit_batch_gamma_dists_to_gamma_mixtures`
*	`rough_fit_batch_gamma_dists_to_gamma_mixtures`

The first function solves a single Gamma mixture
approximation problem, while the batch functions take
a batch of many problems and solves them iteratively.
For an example of usage of the batched fit, see the test.
The API for both functions is defined in the
`lib/gamma_approx/_gammaapprox.pyx` Cython source file.

The `rough_` variant of the `fit_batch_` function uses
a fixed number of iterations, which accelerates performance
but may give lower accuracy results for some inputs.


### Description:

The `gamma_approx` library offers routines to fit a single
Gamma distribution to approximate some given mixture of Gamma
distributions. The parameters of the approximating Gamma
distribution are chosen to minimise the KL-divergence from
the approximation to the original Gamma mixture.

For each fit, the library does the following:

1.	Computes the expected rate and expected log rate
	of the input Gamma mixture.

2.	Obtains a two-dimensional system of nonlinear
	constraints equating the expected rate and expected
	log rate of the approximation with the expected
	rate and expected log rate of the input mixture,
	respectively.

3.	Computes the optimal value of the approximating
	shape parameter by solving a one-dimensional
	problem to invert the function `y = digamma(x) - log(x)`.
	This is perfomed numerically using Halley's
	method, using a series approximation for `digamma`.

4.	Solves for the optimal value of the remaining rate
	parameter in terms of the optimal shape parameter
	and the input data. In constrast to step 3, this is
	immediate.

Note that performance appears to be greatly accelerated when
solving large batches of these approximation problems.

### Performance (single core)

The "rough" lower-accuracy batched implementation
can fit 320k randomly-sized mixtures, each consisting
of 1 -- 20 component Gamma distributions in around 0.106
seconds on an AMD 2200G CPU, using a single core.
That is:

*	331 nanos per mixture fit, or equivalently
*	3 million mixture fits per second.
