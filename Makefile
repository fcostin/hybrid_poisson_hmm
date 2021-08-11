LIBGAMMAAPPROX=lib/gamma_approx/_gammaapprox.cpython-38-x86_64-linux-gnu.so

test:
	python -m pytest -v .
.PHONY: test

lib:    $(LIBGAMMAAPPROX)
.PHONY: lib

$(LIBGAMMAAPPROX): setup.py lib/gamma_approx/_gammaapprox.pyx
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -rf lib/gamma_approx/_gammaapprox.c
	rm -rf lib/gamma_approx/_gammaapprox.*.so
	rm -rf lib/gamma_approx/_gammaapprox.html
.PHONY: clean
