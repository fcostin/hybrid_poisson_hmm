import abc
import numpy.typing

class BaseHMM(abc.ABC):

    @abc.abstractmethod
    def forward(self,
                observations: numpy.typing.NDArray[int],
                q0: numpy.typing.NDArray,
    ) -> numpy.typing.NDArray:
        return NotImplemented