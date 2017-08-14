import numpy as np
import batoid

def parallelRays(z, outer, inner=0, theta_x=0, theta_y=0, nradii=50, naz=64,
                 wavelength=500e-9, medium=batoid.ConstMedium(1.0)):
    """Create a RayVector of parallel Rays aimed toward the origin.

    Parameters
    ----------
    z : float
        distance of center of Ray origins to origin in meters
    outer : float
        Outer radius of distribution of rays in meters
    inner : float
        Inner radius of distribution of rays in meters [default: 0.0]
    theta_x : float
        x-axis field angle in radians [default: 0.0]
    theta_y : float
        y-axis field angle in radians [default: 0.0]
    nradii : int
        Number of radii to use [default: 50]
    naz : int
        Number of azimuthal rays to use on the outer ring. [default: 64]
    wavelength : float
        Wavelength of rays in meters. [default: 500e-9]
    medium : batoid.Medium
        Medium rays are in. [default: batoid.ConstMedium(1.0)]
    """
    radii = np.linspace(inner, outer, nradii)
    rays = batoid.RayVector()
    try:
        n = medium.getN(wavelength)
    except AttributeError:
        n = medium

    # Make a centered ray that can be used to define initial plane wave.
    r0 = batoid.Ray(batoid.Vec3(0, 0, 0),
                    batoid.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1),
                    t=z*n)
    p00 = r0.positionAtTime(0)
    start_plane = batoid.Plane(p00.z).rotX(-theta_y).rotY(theta_x)

    for r in radii:
        phis = np.linspace(0, 2*np.pi, int(naz*r/outer), endpoint=False)
        for phi in phis:
            p0 = batoid.Vec3(r*np.cos(phi), r*np.sin(phi), 0)
            v = batoid.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1)
            v *= 1./(n*v.Magnitude())
            r0 = batoid.Ray(p0, v, t=z*n)
            isec = start_plane.intersect(r0)
            rays.append(batoid.Ray(isec.point, r0.v, t=0, w=wavelength))
    return rays


def rayGrid(z, size, theta_x=0, theta_y=0, nx=64,
            wavelength=500e-9, medium=batoid.ConstMedium(1.0)):
    """Create a RayVector of parallel Rays aimed toward the origin.

    Parameters
    ----------
    z : float
        distance of center of Ray origins to origin in meters
    theta_x : float
        x-axis field angle in radians [default: 0.0]
    theta_y : float
        y-axis field angle in radians [default: 0.0]
    wavelength : float
        Wavelength of rays in meters. [default: 500e-9]
    medium : batoid.Medium
        Medium rays are in. [default: batoid.ConstMedium(1.0)]
    """
    rays = batoid.RayVector()
    try:
        n = medium.getN(wavelength)
    except AttributeError:
        n = medium

    # Make a centered ray that can be used to define initial plane wave.
    r0 = batoid.Ray(batoid.Vec3(0, 0, 0),
                    batoid.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1),
                    t=z*n)
    p00 = r0.positionAtTime(0)
    start_plane = batoid.Plane(p00.z).rotX(-theta_y).rotY(theta_x)

    for y in np.linspace(-size/2, size/2, nx, endpoint=True):
        for x in np.linspace(-size/2, size/2, nx, endpoint=True):
            p0 = batoid.Vec3(x, y, 0)
            v = batoid.Vec3(-np.tan(theta_x), -np.tan(theta_y), -1)
            v *= 1./(n*v.Magnitude())
            r0 = batoid.Ray(p0, v, t=z*n)
            isec = start_plane.intersect(r0)
            rays.append(batoid.Ray(isec.point, r0.v, t=0, w=wavelength))
    return rays