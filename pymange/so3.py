import numpy as np


class SO3:
    _EPSILON = 1e-12

    def __init__(self):
        self._C = np.eye(3)

    @staticmethod
    def fromRotation(rotation):
        if not isinstance(rotation, np.ndarray) and rotation.shape == (3,):
            raise TypeError("rotation must a numpy array with shape (3,)")
        return SO3.Exp(rotation)

    @staticmethod
    def Exp(rotation):
        X = SO3()
        X._Exp(rotation)
        return X

    def _Exp(self, x):
        phi = np.linalg.norm(x)

        if phi > SO3._EPSILON:
            a = x / phi

            self._C = np.cos(phi)*np.eye(3) + \
                (1.0 - np.cos(phi))*np.outer(a, a) + np.sin(phi)*SO3.hat(a)
        else:
            X = SO3.hat(x)
            X2 = X @ X
            X3 = X2 @ X
            X4 = X3 @ X

            self._C = np.eye(3) + X + X2/2 + X3/6 + X4/24

    def Log(self):
        if self.isIdentity():
            return np.zeros((3,))
        else:
            w, v = np.linalg.eig(self._C)
            # find index of eigenvalue closest to 1
            index = (np.abs(w - 1.0)).argmin()
            a = v[:, index]

            phi = np.arccos((np.trace(self._C) - 1) / 2)

            test = SO3.Exp(phi*a)
            if not np.allclose(test._C, self._C):
                phi = -phi

            return phi*a

    def Ad(self):
        return self._C

    @staticmethod
    def Jl(x):
        phi = np.linalg.norm(x)

        if phi > SO3._EPSILON:
            a = x / phi
            c1 = np.sin(phi) / phi
            c2 = 1.0 - c1
            c3 = (1.0 - np.cos(phi)) / phi
            return c1*np.eye(3) + c2*np.outer(a, a) + c3*SO3.hat(a)
        else:
            X = SO3.hat(x)
            X2 = X @ X
            X3 = X2 @ X
            return np.eye(3) + X/2 + X2/6 + X3/24

    @staticmethod
    def Jr(x):
        return SO3.Jl(-x)

    @staticmethod
    def JlInverse(x):
        phi = np.linalg.norm(x)

        multiplier = phi / (2 * np.pi)
        if phi > SO3._EPSILON and abs(multiplier - np.round(multiplier)) > SO3._EPSILON:
            a = x / phi
            c1 = phi/2 * np.cot(phi/2)
            c2 = 1.0 - c1
            c3 = -phi/2
            return c1*np.eye(3) + c2*np.outer(a, a) + c3*SO3.hat(a)
        else:
            X = SO3.hat(x)
            X2 = X @ X
            X4 = X2 @ X2
            return np.eye(3) - X/2 + X2/12 - X4/720


    @staticmethod
    def JrInverse(x):
        return SO3.JlInverse(-x)

    @staticmethod
    def hat(phi):
        return np.array([[0.0, -phi[2], phi[1]],
                         [phi[2], 0.0, -phi[0]],
                         [-phi[1], phi[0], 0.0]])

    @staticmethod
    def vee(x):
        return np.array([x[2, 1], x[0, 2], x[1, 0]])

    def inverse(self):
        result = SO3()
        result._C = self._C.T
        return result

    def __mul__(self, rhs):
        if isinstance(rhs, SO3):
            result = SO3()
            result._C = self._C @ rhs._C
            return result
        elif isinstance(rhs, np.ndarray) and (rhs.shape == (3,) or rhs.shape[-2] == 3):
            return self._C @ rhs
        else:
            raise TypeError(
                "Unsupported operand type for *; must be SO3 or numpy.ndarray compatible with left multiplication by 3x3 array")

    def __matmul__(self, rhs):
        return self.__mul__(rhs)

    def setIdentity(self):
        self._C = np.eye(3)

    def getRotation(self):
        return self.Log()

    def isIdentity(self):
        return np.allclose(self._C, np.eye(3))
