import numpy as np


class SO2:
    def __init__(self):
        self._C = np.eye(2)

    @staticmethod
    def fromRotation(rotation):
        if not isinstance(rotation, float):
            raise TypeError("rotation must be a float")
        return SO2.Exp(rotation)

    @staticmethod
    def Exp(rotation):
        X = SO2()
        X._Exp(rotation)
        return X

    def _Exp(self, rotation):
        self._C = np.array(
            [[np.cos(rotation), -np.sin(rotation)],
             [np.sin(rotation), np.cos(rotation)]])

    def Log(self):
        return np.arctan2(self._C[1, 0], self._C[0, 0])

    def Ad(self):
        return 1.0

    @staticmethod
    def Jl(phi):
        return 1.0

    @staticmethod
    def Jr(phi):
        return 1.0

    @staticmethod
    def JlInverse(phi):
        return 1.0

    @staticmethod
    def JrInverse(phi):
        return 1.0

    @staticmethod
    def hat(phi):
        return np.array([[0.0, -phi], [phi, 0.0]])

    @staticmethod
    def vee(x):
        return x[1,0]

    def inverse(self):
        result = SO2()
        result._C = self._C.T
        return result

    def __mul__(self, rhs):
        if isinstance(rhs, SO2):
            result = SO2()
            result._C = self._C @ rhs._C
            return result
        elif isinstance(rhs, np.ndarray) and (rhs.shape == (2,) or rhs.shape == (2, 1) or rhs.shape == (2, 2)):
            return self._C @ rhs
        else:
            raise TypeError(
                "Unsupported operand type for *; must be SO2 or numpy.ndarray with shape (2,), (2,1) or (2,2)")

    def __matmul__(self, rhs):
        return self.__mul__(rhs)

    def setIdentity(self):
        self._C = np.eye(2)

    def getRotation(self):
        return self.Log()