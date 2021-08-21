LIBGAMMAAPPROX=lib/gamma_approx/_gammaapprox.cpython-38-x86_64-linux-gnu.so
LIBHPHMM=lib/hphmm/libhphmm.cpython-38-x86_64-linux-gnu.so

test:
	python -m pytest -v .
.PHONY: test

libs:    $(LIBGAMMAAPPROX) $(LIBHPHMM)
.PHONY: libs

$(LIBGAMMAAPPROX): setup_gamma_approx.py lib/gamma_approx/_gammaapprox.pyx
	python setup_gamma_approx.py build_ext --inplace

$(LIBHPHMM): setup_hphmm.py lib/hphmm/libhphmm.pyx
	python setup_hphmm.py build_ext --inplace


clean:
	rm -rf build
	rm -rf lib/gamma_approx/_gammaapprox.c
	rm -rf lib/gamma_approx/_gammaapprox.*.so
	rm -rf lib/gamma_approx/_gammaapprox.html
	rm -rf lib/hphmm/libhphmm.c
	rm -rf lib/hphmm/libhphmm.*.so
	rm -rf lib/hphmm/libhphmm.html
.PHONY: clean
