# hybrid discrete/continuous hidden Markov model with Poisson noise

## `gamma_approx` library

### Purpose:

`gamma_approx` is a Python library to approximate mixtures
of Gamma distributions with a single Gamma distribution.

![example-gamma-mixture-plot](./gallery/example-gamma-mixture-approx.png)

### Status

Prototype

### License

BSD

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
solving large batches of these approximation problems. There
must be considerable overhead in the python/cython
wrapper bindings. This could use further investigation.

The batched implementation can fit 320k randomly-sized
mixtures, each consistenting of 1 -- 20 component Gamma
distributions in less than 0.175 seconds on an AMD 2200G
CPU, using a single core. That's a running time of around
540 nanoseconds per mixture fit.
