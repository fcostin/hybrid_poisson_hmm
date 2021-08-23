from .make_problems import make_problem
import numpy
import numpy.random


def main():
    seed=None
    rng = numpy.random.default_rng(seed)
    p = make_problem(rng)
    print(repr(p))

if __name__ == '__main__':
    main()