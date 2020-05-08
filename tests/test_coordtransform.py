import numpy as np
import batoid
from test_helpers import ray_isclose, rays_allclose, timer, do_pickle


def randomVec3():
    import random
    return np.array([
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1)
    ])


def randomCoordSys():
    import random
    return batoid.CoordSys(
        randomVec3(),
        (batoid.RotX(random.uniform(0, 1))
         .dot(batoid.RotY(random.uniform(0, 1)))
         .dot(batoid.RotZ(random.uniform(0, 1))))
    )


def randomRay():
    import random
    return batoid.Ray(
        randomVec3(),
        randomVec3(),
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1),
    )


def randomRayVector():
    import random
    return batoid.RayVector(
        [randomRay() for i in range(10)]
    )


@timer
def test_composition():
    import random
    random.seed(5)

    for i in range(10):
        coordSys1 = randomCoordSys()
        coordSys2 = randomCoordSys()
        coordSys3 = randomCoordSys()

        assert coordSys1 != coordSys2
        assert coordSys1 != coordSys3

        do_pickle(coordSys1)

        transform1to2 = batoid.CoordTransform(coordSys1, coordSys2)
        transform1to3 = batoid.CoordTransform(coordSys1, coordSys3)
        transform2to3 = batoid.CoordTransform(coordSys2, coordSys3)

        do_pickle(transform1to2)

        for i in range(10):
            vec3 = randomVec3()
            vec3_a = transform1to3.applyForward(vec3.copy())
            vec3_b = transform2to3.applyForward(transform1to2.applyForward(vec3.copy()))
            np.testing.assert_allclose(vec3_a, vec3_b)

            vec3_ra = transform1to3.applyReverse(vec3.copy())
            vec3_rb = transform1to2.applyReverse(transform2to3.applyReverse(vec3.copy()))
            np.testing.assert_allclose(vec3_ra, vec3_rb)

            ray = randomRay()
            ray_a = transform1to3.applyForward(ray.copy())
            assert ray_a != ray
            ray_b = transform2to3.applyForward(transform1to2.applyForward(ray.copy()))
            assert ray_b != ray
            assert ray_isclose(ray_a, ray_b), "error with composite transform of Ray"
            ray_ra = transform1to3.applyReverse(ray.copy())
            ray_rb = transform1to2.applyReverse(transform2to3.applyReverse(ray.copy()))
            assert ray_isclose(ray_ra, ray_rb), "error with reverse composite transform of Ray"

            rv = randomRayVector()
            rv_a = transform1to3.applyForward(rv.copy())
            assert rv_a != rv
            rv_b = transform2to3.applyForward(transform1to2.applyForward(rv.copy()))
            assert rv_b != rv
            assert rays_allclose(rv_a, rv_b), "error with composite transform of RayVector"
            rv_ra = transform1to3.applyReverse(rv.copy())
            rv_rb = transform1to2.applyReverse(transform2to3.applyReverse(rv.copy()))
            assert rays_allclose(rv_ra, rv_rb), "error with reverse composite transform of RayVector"

            # Test with numpy arrays
            xyz = rv.r.T
            xyz_a = transform1to3.applyForward(*xyz.copy())
            xyz_b = transform2to3.applyForward(*transform1to2.applyForward(*xyz.copy()))
            xyz_c = [transform2to3.applyForward(transform1to2.applyForward(r.r)) for r in rv]
            assert xyz_a[0].ctypes.data != xyz_b[0].ctypes.data
            assert xyz_b[0].ctypes.data != xyz_c[0].ctypes.data
            np.testing.assert_allclose(xyz_a, xyz_b)
            np.testing.assert_allclose(xyz_a, np.transpose(xyz_c))
            # Should still work if we reshape.
            xyz2 = xyz.reshape((3, 2, 5))
            xyz2_a = transform1to3.applyForward(*xyz2.copy())
            xyz2_b = transform2to3.applyForward(*transform1to2.applyForward(*xyz2.copy()))
            np.testing.assert_allclose(xyz2_a, xyz2_b)

            # And also work if we reverse
            np.testing.assert_allclose(xyz, transform1to3.applyReverse(*xyz_a))
            np.testing.assert_allclose(xyz, transform1to2.applyReverse(*transform2to3.applyReverse(*xyz_b)))

            # Test in-place on Ray
            ray = randomRay()
            ray_copy = ray.copy()
            transform1to2.applyForward(ray)
            transform2to3.applyForward(ray)
            transform1to3.applyForward(ray_copy)
            assert ray_isclose(ray, ray_copy)

            # in-place reverse on Ray
            ray = randomRay()
            ray_copy = ray.copy()
            transform2to3.applyReverse(ray)
            transform1to2.applyReverse(ray)
            transform1to3.applyReverse(ray_copy)
            assert ray_isclose(ray, ray_copy)

            # Test in-place on RayVector
            rv = randomRayVector()
            rv_copy = rv.copy()
            transform1to2.applyForward(rv)
            transform2to3.applyForward(rv)
            transform1to3.applyForward(rv_copy)
            assert rays_allclose(rv, rv_copy)

            # in-place reverse on RayVector
            rv = randomRayVector()
            rv_copy = rv.copy()
            transform2to3.applyReverse(rv)
            transform1to2.applyReverse(rv)
            transform1to3.applyReverse(rv_copy)
            assert rays_allclose(rv, rv_copy)

            # Check on exceptions
            with np.testing.assert_raises(RuntimeError):
                transform1to2.applyForward(np.array([1., 2, 3, 4]))
            with np.testing.assert_raises(RuntimeError):
                transform1to2.applyForward(np.array([1., 2]))
            with np.testing.assert_raises(RuntimeError):
                transform1to2.applyForward(
                    np.array([1, 2, 3, 4.]),
                    np.array([1, 2, 3, 4.]),
                    np.array([1, 2, 3.])
                )
            with np.testing.assert_raises(TypeError):
                transform1to2.applyForward(1, 2, 3.)



if __name__ == '__main__':
    test_composition()
