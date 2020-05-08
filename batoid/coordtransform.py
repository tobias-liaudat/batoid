from . import _batoid
import numpy as np


class CoordTransform:
    """Transformation between two coordinate systems.

    Parameters
    ----------
    fromSys : CoordSys
        Origin coordinate systems.
    toSys : CoordSys
        Destination coordinate systems.
    """
    def __init__(self, fromSys, toSys):
        self.fromSys = fromSys
        self.toSys = toSys
        self._coordTransform = _batoid.CPPCoordTransform(
            fromSys._coordSys,
            toSys._coordSys
        )

    def applyForward(self, arg1, arg2=None, arg3=None):
        """Apply forward-direction transformation in place.

        Parameters
        ----------
        arg : Ray or RayVector or array of float
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector or array
            Reference to transformed input.

        Notes
        -----
        Transformations are applied in place!  An implication is that numpy
        arrays being transformed must be writeable have dtype float to contain
        the output.
        """
        from .ray import Ray
        from .rayVector import RayVector
        if arg2 is not None:  # numpy arrays to transform (not-in-place)
            self._coordTransform.applyForward(arg1, arg2, arg3)
            return arg1, arg2, arg3
        elif isinstance(arg1, np.ndarray):  # single np array
            self._coordTransform.applyForward(arg1)
            return arg1
        elif isinstance(arg1, (Ray, RayVector)):
            self._coordTransform.applyForward(arg1._rv)
            return arg1
        else:
            raise TypeError(f"Invalid type {type(arg1)}")

    def applyReverse(self, arg1, arg2=None, arg3=None):
        """Apply reverse-direction transformation in place.

        Parameters
        ----------
        arg : Ray or RayVector or array of float
            Object to transform.  Return type is the same as the input type.

        Returns
        -------
        transformed : Ray or RayVector or array
            Reference to transformed input.

        Notes
        -----
        Transformations are applied in place!  An implication is that numpy
        arrays being transformed must be writeable have dtype float to contain
        the output.
        """
        from .ray import Ray
        from .rayVector import RayVector
        if arg2 is not None:  # numpy arrays to transform (not-in-place)
            self._coordTransform.applyReverse(arg1, arg2, arg3)
            return arg1, arg2, arg3
        elif isinstance(arg1, np.ndarray):  # single np array
            self._coordTransform.applyReverse(arg1)
            return arg1
        elif isinstance(arg1, (Ray, RayVector)):
            self._coordTransform.applyReverse(arg1._rv)
            return arg1
        else:
            raise TypeError(f"Invalid type {type(arg1)}")

    def __eq__(self, rhs):
        if not isinstance(rhs, CoordTransform): return False
        return self._coordTransform == rhs._coordTransform

    def __ne__(self, rhs):
        return not (self == rhs)

    def __repr__(self):
        return repr(self._coordTransform)

    def __hash__(self):
        return hash(self._coordTransform)
