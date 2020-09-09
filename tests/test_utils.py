import batoid
from test_helpers import timer

import numpy as np


@timer
def test_normalized():
    rng = np.random.default_rng(5)
    for _ in range(1000):
        x = rng.uniform()
        y = rng.uniform()
        z = rng.uniform()
        w = rng.uniform()

        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y, z])),
            1.0,
            rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(batoid.utils.normalized([x, y, z, w])),
            1.0,
            rtol=0, atol=1e-10
        )


@timer
def test_gnomonicDirCos():
    rng = np.random.default_rng(57)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToGnomonic(*batoid.utils.gnomonicToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.1, 0.1, size=10000)
    beta = rng.uniform(-0.1, 0.1, size=10000)
    gamma = -np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.gnomonicToDirCos(
        *batoid.utils.dirCosToGnomonic(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.gnomonicToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_postelDirCos():
    rng = np.random.default_rng(577)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToPostel(*batoid.utils.postelToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.1, 0.1, size=10000)
    beta = rng.uniform(-0.1, 0.1, size=10000)
    gamma = -np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.postelToDirCos(
        *batoid.utils.dirCosToPostel(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.postelToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_zemaxDirCos():
    rng = np.random.default_rng(5772)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToZemax(*batoid.utils.zemaxToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.1, 0.1, size=10000)
    beta = rng.uniform(-0.1, 0.1, size=10000)
    gamma = -np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.zemaxToDirCos(
        *batoid.utils.dirCosToZemax(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.zemaxToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_stereographicDirCos():
    rng = np.random.default_rng(57721)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToStereographic(
        *batoid.utils.stereographicToDirCos(u, v)
    )
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.1, 0.1, size=10000)
    beta = rng.uniform(-0.1, 0.1, size=10000)
    gamma = np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.stereographicToDirCos(
        *batoid.utils.dirCosToStereographic(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.stereographicToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_orthographicDirCos():
    rng = np.random.default_rng(577215)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToOrthographic(
        *batoid.utils.orthographicToDirCos(u, v)
    )
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.1, 0.1, size=10000)
    beta = rng.uniform(-0.1, 0.1, size=10000)
    gamma = -np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.orthographicToDirCos(
        *batoid.utils.dirCosToOrthographic(alpha, beta, gamma)
    )
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.orthographicToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_lambertDirCos():
    rng = np.random.default_rng(5772156)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    # Test round trip
    u1, v1 = batoid.utils.dirCosToLambert(*batoid.utils.lambertToDirCos(u, v))
    np.testing.assert_allclose(u, u1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v1, rtol=1e-10, atol=1e-12)

    # Test round trip in the other direction
    alpha = rng.uniform(-0.5, 0.5, size=10000)
    beta = rng.uniform(-0.5, 0.5, size=10000)
    gamma = -np.sqrt(1 - alpha**2 - beta**2)
    alpha1, beta1, gamma1 = batoid.utils.lambertToDirCos(
        *batoid.utils.dirCosToLambert(alpha, beta, gamma)
    )
    # Not sure why Lambert isn't as good as other projections in this test.
    np.testing.assert_allclose(alpha, alpha1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(beta, beta1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gamma, gamma1, rtol=1e-10, atol=1e-12)

    # For really tiny angles, u/v should be basically the same as alpha/beta
    u = rng.uniform(-1e-6, 1e-6, size=10000)
    v = rng.uniform(-1e-6, 1e-6, size=10000)
    alpha, beta, gamma = batoid.utils.lambertToDirCos(u, v)
    np.testing.assert_allclose(alpha, u, rtol=0, atol=1e-8)
    np.testing.assert_allclose(beta, v, rtol=0, atol=1e-8)

    # Check normalization of direction cosines
    np.testing.assert_allclose(
        np.sqrt(alpha*alpha + beta*beta + gamma*gamma),
        1,
        rtol=0, atol=1e-15
    )


@timer
def test_coord():
    rng = np.random.default_rng(57721566)
    import coord
    pole = coord.CelestialCoord(0.*coord.degrees, 90.*coord.degrees)
    u = rng.uniform(-0.5, 0.5, size=10000)
    v = rng.uniform(-0.5, 0.5, size=10000)

    for projection in ['gnomonic', 'stereographic', 'postel', 'lambert']:
        ra, dec = pole.deproject_rad(u, v, projection=projection)
        xcos, ycos, zcos = batoid.utils.fieldToDirCos(
            u, v, projection=projection
        )
        np.testing.assert_allclose(-np.sin(dec), zcos, rtol=0, atol=1e-13)
        np.testing.assert_allclose(
            np.abs((np.pi/2-ra)-np.arctan2(ycos, xcos)),
            np.pi,
            rtol=0, atol=1e-13
        )


if __name__ == '__main__':
    test_normalized()
    test_gnomonicDirCos()
    test_postelDirCos()
    test_zemaxDirCos()
    test_stereographicDirCos()
    test_orthographicDirCos()
    test_lambertDirCos()
    test_coord()