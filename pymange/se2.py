import numpy as np

from .so2 import SO2


class SE2:
    _EPSILON = 1e-12

    def __init__(self):
        self._C = SO2()
        self._r = np.zeros(2)

    @staticmethod
    def fromTranslationAndRotation(translation, rotation):
        if not isinstance(translation, np.ndarray) or not (translation.shape == (2,) or translation.shape == (2, 1)):
            raise TypeError("translation must be a numpy.ndarray with shape (2,) or (2,1)")
        if not isinstance(rotation, float):
            raise TypeError("rotation must be a float")

        X = SE2()
        X._r = translation
        X._C = SO2.fromRotation(rotation)
        return X

    @staticmethod
    def Exp(xi):
        X = SE2()
        X._Exp(xi)
        return X

    def _Exp(self, xi):
        if not isinstance(xi, np.ndarray) or not (xi.shape == (3,) or xi.shape == (3, 1)):
            raise TypeError("xi must be a numpy.ndarray with shape (3,) or (3,1)")

        phi = xi[2]

        self._C = SO2.Exp(phi)
        if abs(phi) > SE2._EPSILON:
            a = np.sin(phi) / phi
            b = (1 - np.cos(phi)) / phi
        else:
            a = 1 - (phi**2)/6 + (phi**4)/120
            b = phi/2 - (phi**3)/24

        self._r = (a * np.eye(2) + b * SO2.hat(1)) @ xi[:2]

    def Log(self):
        phi = self._C.Log()

        if abs(phi) > SE2._EPSILON:
            a = np.sin(phi) / phi
            b = (1 - np.cos(phi)) / phi
        else:
            a = 1 - (phi**2)/6 + (phi**4)/120
            b = phi/2 - (phi**3)/24

        xi = np.empty(3)
        xi[:2] = 1/(a**2 + b**2) * \
            (a * np.eye(2) - b * SO2.hat(1)) @ self._r
        xi[2] = phi
        return xi

    def Ad(self):
        Adj = np.eye(3)
        Adj[:2, :2] = self._C._C
        Adj[0, 2] = self._r[1]
        Adj[1, 2] = -self._r[0]
        return Adj

    @staticmethod
    def Jl(xi):
        if not isinstance(xi, np.ndarray) or not (xi.shape == (3,) or xi.shape == (3, 1)):
            raise TypeError(
                "xi must be a numpy.ndarray with shape (3,) or (3,1)")

        phi = xi[2]
        if abs(phi) > SE2._EPSILON:
            alpha1 = (1 - np.cos(phi)) / phi**2
            alpha2 = (phi - np.sin(phi)) / phi**3
        else:
            alpha1 = 1/2 - (phi**2)/24 + (phi**4)/720 - (phi**6)/40320
            alpha2 = 1/6 - (phi**2)/120 + (phi**4)/5040 - (phi**6)/362880

        ad_xi = SE2._ad(xi)
        return np.eye(3) + alpha1*ad_xi + alpha2*(ad_xi @ ad_xi)

    @staticmethod
    def Jr(xi):
        return SE2.Jl(-xi)

    @staticmethod
    def JlInverse(xi):
        if not isinstance(xi, np.ndarray) or not (xi.shape == (3,) or xi.shape == (3, 1)):
            raise TypeError(
                "xi must be a numpy.ndarray with shape (3,) or (3,1)")

        phi = xi[2]
        while phi > np.pi:
            phi -= 2*np.pi
        while phi < -np.pi:
            phi += 2*np.pi

        if abs(phi) > SE2._EPSILON:
            alpha = 1/(phi**2) - np.cos(phi/2) / (2*phi*np.sin(phi/2))
        else:
            alpha = 1/12 + (phi**2)/720 + (phi**4)/30240 + (phi**6)/1209600

        ad_xi = SE2._ad(xi)
        return np.eye(3) - 0.5*ad_xi + alpha*(ad_xi @ ad_xi)

    @staticmethod
    def JrInverse(xi):
        return SE2.JlInverse(-xi)

    @staticmethod
    def hat(xi):
        if not isinstance(xi, np.ndarray) or not (xi.shape == (3,) or xi.shape == (3, 1)):
            raise TypeError(
                "xi must be a numpy.ndarray with shape (3,) or (3,1)")

        x = np.eye(3)
        x[:2,:2] = SO2.hat(xi[2])
        x[:2,2] = xi[:2]
        return x

    @staticmethod
    def vee(x):
        if not isinstance(x, np.ndarray) or x.shape != (3, 3):
            raise TypeError("x must be a numpy.ndarray with shape (3,3)")

        xi = np.empty((3,1))
        xi[:2] = x[:2,2]
        xi[2] = x[2,2]
        return xi

    def inverse(self):
        result = SE2()
        result._C = self._C.inverse()
        result._r = -(self._C.inverse() @ self._r)
        return result

    def __mul__(self, rhs):
        if isinstance(rhs, SE2):
            result = SE2()
            result._C = self._C * rhs._C
            result._r = self._C * rhs._r + self._r
            return result
        elif isinstance(rhs, np.ndarray) and rhs.shape == (2,):
            return self._C @ rhs + self._r
        elif isinstance(rhs, np.ndarray) and rhs.ndim > 1 and rhs.shape[-2] == 2:
            return self._C @ rhs + self._r[:, np.newaxis]
        else:
            raise TypeError(
                "Unsupported operand type for *; must be SE2 or numpy.ndarray compatible with left multiplication by 2x2 array")

    def __matmul__(self, rhs):
        return self.__mul__(rhs)

    def setIdentity(self):
        self._C.setIdentity()
        self._r = np.zeros((2,1))

    def getTranslationAndRotation(self):
        return self._r, self._C.getRotation()

    @staticmethod
    def _ad(xi):
        ad = np.zeros((3, 3))
        ad[:2, :2] = SO2.hat(xi[2])
        ad[0, 2] = xi[1]
        ad[1, 2] = -xi[0]
        return ad
