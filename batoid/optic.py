import batoid
import numpy as np

class Optic(object):
    """This is the most general category of batoid optical system.  It can include
    a single reflective or refractive surface, a lens consisting of two surfaces, or
    an entire telescope including multiple mirrors and/or surfaces.

    Parameters
    ----------
        name : str or None
            An optional name for this optic

        inMedium : batoid.Medium
            Medium in which approaching rays reside

        outMedium : batoid.Medium
            Medium in which rays will leave this optic

        coordSys : batoid.CoordSys
            Coordinate system indicating the position and orientation of this optic's vertex
            with respect to it's parent's coordinate system.  Optics can be nested, i.e., a camera
            Optic may contain several lens Optics, and may itself be part of a larger telescope
            Optic.
    """
    def __init__(self, name=None, coordSys=batoid.CoordSys(), inMedium=batoid.ConstMedium(1.0),
                 outMedium=batoid.ConstMedium(1.0)):
        self.name = name
        self.inMedium = inMedium
        self.outMedium = outMedium
        if coordSys is None:
            raise ValueError("coordSys required for optic")
        self.coordSys = coordSys

    def _repr_helper(self):
        out = ""
        if self.name is not None:
            out += ", name={!r}".format(self.name)
        out += ", coordSys={!r}".format(self.coordSys)
        if self.inMedium != batoid.ConstMedium(1.0):
            out += ", inMedium={!r}".format(self.inMedium)
        if self.outMedium != batoid.ConstMedium(1.0):
            out += ", outMedium={!r}".format(self.outMedium)
        return out



class Interface(Optic):
    """The most basic category of Optic representing a single surface.  Almost always one of the
    concrete subclasses should be instantiated, depending on whether rays should reflect, refract,
    or simply pass through this surface.

    Parameters
    ----------
        surface : batoid.Surface
            The surface instance for this Interface.

        obscuration : batoid.Obscuration or None
            Batoid.Obscuration instance indicating which x,y coordinates are obscured/unobscured
            for rays intersecting this optic.

        **kwargs :
            Other parameters to forward to Optic.__init__.
    """
    def __init__(self, surface, obscuration=None, **kwargs):
        Optic.__init__(self, **kwargs)

        self.surface = surface
        self.obscuration = obscuration

        # Stealing inRadius and outRadius from self.obscuration.  These are required for the draw
        # method.  Only works at the moment if obscuration is a negated circle.  Should at least
        # extend this to negated annulus.
        self.inRadius = 0.0
        self.outRadius = None
        if self.obscuration is not None:
            if isinstance(self.obscuration, batoid.ObscNegation):
                if isinstance(self.obscuration.original, batoid.ObscCircle):
                    self.outRadius = self.obscuration.original.radius
                elif isinstance(self.obscuration.original, batoid.ObscAnnulus):
                    self.outRadius = self.obscuration.original.outer
                    self.inRadius = self.obscuration.original.inner

    def __hash__(self):
        return hash((self.__class__.__name__, self.surface, self.obscuration, self.name,
                     self.inMedium, self.outMedium, self.coordSys))

    def draw(self, ax):
        """ Draw this interface on a mplot3d axis.

        Parameters
        ----------

            ax : mplot3d.Axis
                Axis on which to draw this optic.
        """
        if self.outRadius is None:
            return
        # Going to draw 4 objects here: inner circle, outer circle, sag along x=0, sag along y=0
        # inner circle
        if self.inRadius != 0.0:
            th = np.linspace(0, 2*np.pi, 100)
            cth, sth = np.cos(th), np.sin(th)
            x = self.inRadius * cth
            y = self.inRadius * sth
            z = self.surface.sag(x, y)
            # ax.plot(x, y, z, c='k')
            transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
            x, y, z = transform.applyForward(x, y, z)
            ax.plot(x, y, z, c='k')

        #outer circle
        th = np.linspace(0, 2*np.pi, 100)
        cth, sth = np.cos(th), np.sin(th)
        x = self.outRadius * cth
        y = self.outRadius * sth
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

        #next, a line at X=0
        y = np.linspace(-self.outRadius, -self.inRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')
        y = np.linspace(self.inRadius, self.outRadius)
        x = np.zeros_like(y)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

        #next, a line at Y=0
        x = np.linspace(-self.outRadius, -self.inRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')
        x = np.linspace(self.inRadius, self.outRadius)
        y = np.zeros_like(x)
        z = self.surface.sag(x, y)
        transform = batoid.CoordTransform(self.coordSys, batoid.CoordSys())
        x, y, z = transform.applyForward(x, y, z)
        ax.plot(x, y, z, c='k')

    def trace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Trace a ray through this optical element.

        Parameters
        ----------

        r : batoid.Ray or batoid.RayVector
            input ray to trace

        inCoordSys : batoid.CoordSys
            Coordinate system of incoming ray(s).  Default: the global coordinate system.

        outCoordSys : batoid.CoordSys
            Coordinate system into which to project the output ray(s).  Default: None,
            which means use the coordinate system of the optic.

        Returns
        -------
            Ray or RayVector, output CoordSys.

        """
        transform = batoid.CoordTransform(inCoordSys, self.coordSys)
        r = transform.applyForward(r)
        r = self.surface.intercept(r)
        if self.obscuration is not None:
            r = self.obscuration.obscure(r)
        # refract, reflect, passthrough, depending on subclass
        r = self.interact(r)
        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid.CoordTransform(self.coordSys, outCoordSys)
            return transform.applyForward(r), outCoordSys

    def traceFull(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        result = [{'name':self.name, 'in':r, 'inCoordSys':inCoordSys}]
        r, outCoordSys = self.trace(r, inCoordSys=inCoordSys, outCoordSys=outCoordSys)
        result[0]['out'] = r
        result[0]['outCoordSys'] = outCoordSys
        return result

    def traceInPlace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        transform = batoid.CoordTransform(inCoordSys, self.coordSys)
        transform.applyForwardInPlace(r)
        self.surface.interceptInPlace(r)
        if self.obscuration is not None:
            self.obscuration.obscureInPlace(r)
        # refract, reflect, passthrough, depending on subclass
        self.interactInPlace(r)
        if outCoordSys is None:
            return r, self.coordSys
        else:
            transform = batoid.CoordTransform(self.coordSys, outCoordSys)
            transform.applyForwardInPlace(r)
            return r, outCoordSys

    def printCoordSys(self):
        print(self.name, self.coordSys)

    def printMedium(self):
        print(self.name, self.inMedium, self.outMedium)

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return (self.surface == other.surface and
                self.obscuration == other.obscuration and
                self.name == other.name and
                self.inMedium == other.inMedium and
                self.outMedium == other.outMedium and
                self.coordSys == other.coordSys)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        out = "{!s}({!r}".format(self.__class__.__name__, self.surface)
        if self.obscuration is not None:
            out += ", obscuration={!r}".format(self.obscuration)
        out += Optic._repr_helper(self)
        out +=")"
        return out


class RefractiveInterface(Interface):
    """Specialization for refractive interfaces.
    """
    def interact(self, r):
        return batoid._batoid.refract(r, self.surface, self.inMedium, self.outMedium)

    def interactInPlace(self, r):
        batoid._batoid.refractInPlace(r, self.surface, self.inMedium, self.outMedium)


class Mirror(Interface):
    """Specialization for reflective interfaces.
    """
    def interact(self, r):
        return batoid._batoid.reflect(r, self.surface)

    def interactInPlace(self, r):
        batoid._batoid.reflectInPlace(r, self.surface)


class Detector(Interface):
    """Specialization for detector interfaces.
    """
    def interact(self, r):
        return r

    def interactInPlace(self, r):
        pass


class Baffle(Interface):
    """Specialization for baffle interfaces.  Rays will be reported here, but pass through in
    straight lines.
    """
    def interact(self, r):
        return r

    def interactInPlace(self, r):
        pass


class CompoundOptic(Optic):
    """A Optic containing two or more Optics as subitems.
    """
    def __init__(self, items, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)

    def trace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Recursively trace through this Optic by successively tracing through all subitems.
        """
        coordSys = inCoordSys
        for item in self.items[:-1]:
            r, coordSys = item.trace(r, inCoordSys=coordSys)
        return self.items[-1].trace(r, inCoordSys=coordSys, outCoordSys=outCoordSys)

    def traceInPlace(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        coordSys = inCoordSys
        for item in self.items[:-1]:
            r, coordSys = item.traceInPlace(r, inCoordSys=coordSys)
        return self.items[-1].traceInPlace(r, inCoordSys=coordSys, outCoordSys=outCoordSys)

    def traceFull(self, r, inCoordSys=batoid.CoordSys(), outCoordSys=None):
        """ Recursively trace through this Optic by successively tracing through all subitems.

        The return value will contain the incoming and outgoing rays for each Interface.
        """
        result = []
        r_in = r
        coordSys = inCoordSys
        for item in self.items[:-1]:
            result.extend(item.traceFull(r_in, inCoordSys=coordSys))
            r_in = result[-1]['out']
            coordSys = result[-1]['outCoordSys']
        result.extend(self.items[-1].traceFull(r_in, inCoordSys=coordSys, outCoordSys=outCoordSys))
        return result

    def draw(self, ax):
        """ Recursively draw this Optic by successively drawing all subitems.
        """
        for item in self.items:
            item.draw(ax)

    def printCoordSys(self):
        for item in self.items:
            item.printCoordSys()

    def printMedium(self):
        for item in self.items:
            item.printMedium()

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return (self.items == other.items and
                self.name == other.name and
                self.inMedium == other.inMedium and
                self.outMedium == other.outMedium and
                self.coordSys == other.coordSys)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        out = "{!s}([".format(self.__class__.__name__)
        for item in self.items[:-1]:
            out += "{!r}, ".format(item)
        out += "{!r}]".format(self.items[-1])
        out += Optic._repr_helper(self)
        out += ")"
        return out

    def __hash__(self):
        return hash((self.__class__.__name__, self.items,
                     self.name, self.inMedium, self.outMedium, self.coordSys))


class Lens(CompoundOptic):
    def __init__(self, items, medium, **kwargs):
        Optic.__init__(self, **kwargs)
        self.items = tuple(items)
        self.medium = medium

    def __eq__(self, other):
        if not CompoundOptic.__eq__(self, other):
            return False
        return self.medium == other.medium

    def __repr__(self):
        out = ("{!s}([{!r}, {!r}], {!r}"
               .format(self.__class__.__name__, self.items[0], self.items[1], self.medium))
        out += Optic._repr_helper(self)
        out += ")"
        return out

    def __hash__(self):
        return hash((self.medium, CompoundOptic.__hash__(self)))


# Should pythonize RayVector so can identify when all the wavelengths are the same
# or not, and elide medium.getN() in the inner loop when possible.

# Generic trace looks like
# 1) Change coords from previous element to current local coords.
# 2) Find intersections
# 3) Mark vignetted rays
# 4) Compute reflection/refraction.
#
# Maybe start by converting global -> local -> global for every interface.  Then try
# a more optimized approach?
