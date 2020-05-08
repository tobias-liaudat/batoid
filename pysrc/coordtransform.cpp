#include "coordtransform.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <tuple>
#include <iostream>

namespace py = pybind11;

namespace batoid {
    // Version of applyForward that mutates three congruent numpy arrays (x, y, z) in place
    void numpyApplyForward(
        const CoordTransform& ct,
        py::array_t<double>& xs,
        py::array_t<double>& ys,
        py::array_t<double>& zs)
    {
        auto bufX = xs.request();
        auto bufY = ys.request();
        auto bufZ = zs.request();
        if (bufX.ndim != bufY.ndim || bufX.ndim != bufZ.ndim)
            throw std::runtime_error("Dimensions must match");
        if (bufX.size != bufY.size || bufX.size != bufZ.size)
            throw std::runtime_error("Sizes much match");
        if (!(xs.writeable() && ys.writeable() && zs.writeable()))
            throw std::runtime_error("Arrays must be writeable");

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(bufX.ndim, 0);
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            double* ptrX = (double *) bufX.ptr;
            double* ptrY = (double *) bufY.ptr;
            double* ptrZ = (double *) bufZ.ptr;
            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                ptrX += idxVec[idim]*(bufX.strides[idim]/sizeof(double));
                ptrY += idxVec[idim]*(bufY.strides[idim]/sizeof(double));
                ptrZ += idxVec[idim]*(bufZ.strides[idim]/sizeof(double));
            }

            v = Vector3d(*ptrX, *ptrY, *ptrZ);
            ct.applyForward(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];

            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == bufX.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
    }


    // Version of applyReverse that mutates three congruent numpy arrays (x, y, z) in place
    void numpyApplyReverse(
        const CoordTransform& ct,
        py::array_t<double>& xs,
        py::array_t<double>& ys,
        py::array_t<double>& zs)
    {
        auto bufX = xs.request();
        auto bufY = ys.request();
        auto bufZ = zs.request();
        if (bufX.ndim != bufY.ndim || bufX.ndim != bufZ.ndim)
            throw std::runtime_error("Dimensions must match");
        if (bufX.size != bufY.size || bufX.size != bufZ.size)
            throw std::runtime_error("Sizes much match");
        if (!(xs.writeable() && ys.writeable() && zs.writeable()))
            throw std::runtime_error("Arrays must be writeable");

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(bufX.ndim, 0);
        for (ssize_t idx = 0; idx < bufX.size; idx++) {
            double* ptrX = (double *) bufX.ptr;
            double* ptrY = (double *) bufY.ptr;
            double* ptrZ = (double *) bufZ.ptr;
            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                ptrX += idxVec[idim]*(bufX.strides[idim]/sizeof(double));
                ptrY += idxVec[idim]*(bufY.strides[idim]/sizeof(double));
                ptrZ += idxVec[idim]*(bufZ.strides[idim]/sizeof(double));
            }

            v = Vector3d(*ptrX, *ptrY, *ptrZ);
            ct.applyReverse(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];

            for (auto idim=bufX.ndim-1; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == bufX.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
    }


    // Version of applyForward that mutates single numpy array with trailing dimension 3 in place.
    void numpyApplyForward(
        const CoordTransform& ct,
        py::array_t<double>& rs)
    {
        auto buf = rs.request();
        auto ndim = buf.ndim;
        if (buf.shape[ndim-1] != 3)
            throw std::runtime_error("Trailing dimension must be 3");
        if (!rs.writeable())
            throw std::runtime_error("Array must be writeable");

        // Special case if we've been given a 1-dimensional array
        if (ndim == 1) {
            double* ptrX = ((double *) buf.ptr);
            double* ptrY = ((double *) buf.ptr) + buf.strides[0]/sizeof(double);
            double* ptrZ = ((double *) buf.ptr) + 2*buf.strides[0]/sizeof(double);
            Vector3d v(*ptrX, *ptrY, *ptrZ);
            ct.applyForward(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];
            return;
        }

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(ndim-1, 0);
        int dptr = buf.strides[ndim-1]/sizeof(double);
        for (ssize_t idx = 0; idx < buf.size/3; idx++) {
            double* ptrX = (double *) buf.ptr;
            double* ptrY = ptrX + dptr;
            double* ptrZ = ptrY + dptr;
            for (auto idim=buf.ndim-2; idim >= 0; idim--) {
                ptrX += idxVec[idim]*(buf.strides[idim]/sizeof(double));
                ptrY += idxVec[idim]*(buf.strides[idim]/sizeof(double));
                ptrZ += idxVec[idim]*(buf.strides[idim]/sizeof(double));
            }

            v = Vector3d(*ptrX, *ptrY, *ptrZ);
            ct.applyForward(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];

            for (auto idim=buf.ndim-2; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == buf.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
    }


    // Version of applyReverse that mutates single numpy array with trailing dimension 3 in place.
    void numpyApplyReverse(
        const CoordTransform& ct,
        py::array_t<double>& rs)
    {
        auto buf = rs.request();
        auto ndim = buf.ndim;
        if (buf.shape[ndim-1] != 3)
            throw std::runtime_error("Trailing dimension must be 3");
        if (!rs.writeable())
            throw std::runtime_error("Array must be writeable");

        // Special case if we've been given a 1-dimensional array
        if (ndim == 1) {
            double* ptrX = ((double *) buf.ptr);
            double* ptrY = ((double *) buf.ptr) + buf.strides[0]/sizeof(double);
            double* ptrZ = ((double *) buf.ptr) + 2*buf.strides[0]/sizeof(double);
            Vector3d v(*ptrX, *ptrY, *ptrZ);
            ct.applyReverse(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];
            return;
        }

        // note the mixed radix counting ...
        Vector3d v;
        std::vector<unsigned int> idxVec(ndim-1, 0);
        int dptr = buf.strides[ndim-1]/sizeof(double);
        for (ssize_t idx = 0; idx < buf.size/3; idx++) {
            double* ptrX = (double *) buf.ptr;
            double* ptrY = ptrX + dptr;
            double* ptrZ = ptrY + dptr;
            for (auto idim=buf.ndim-2; idim >= 0; idim--) {
                ptrX += idxVec[idim]*(buf.strides[idim]/sizeof(double));
                ptrY += idxVec[idim]*(buf.strides[idim]/sizeof(double));
                ptrZ += idxVec[idim]*(buf.strides[idim]/sizeof(double));
            }

            v = Vector3d(*ptrX, *ptrY, *ptrZ);
            ct.applyReverse(v);
            *ptrX = v[0];
            *ptrY = v[1];
            *ptrZ = v[2];

            for (auto idim=buf.ndim-2; idim >= 0; idim--) {
                idxVec[idim]++;
                if (idxVec[idim] == buf.shape[idim])
                    idxVec[idim] = 0;
                else
                    break;
            }
        }
    }


    void pyExportCoordTransform(py::module& m) {
        py::class_<CoordTransform, std::shared_ptr<CoordTransform>>(m, "CPPCoordTransform")
            .def(py::init<const CoordSys&, const CoordSys&>())

            // Versions that accept array with trailing dimension 3
            .def(
                "applyForward",
                [](
                    const CoordTransform& ct,
                    py::array_t<double>& rs
                ){
                    numpyApplyForward(ct, rs);
                },
                py::arg().noconvert()
            )
            .def(
                "applyReverse",
                [](
                    const CoordTransform& ct,
                    py::array_t<double>& rs
                ){
                    numpyApplyReverse(ct, rs);
                },
                py::arg().noconvert()
            )

            //Versions that accept 3 congruent arrays of arbitrary dimension.
            .def(
                "applyForward",
                [](
                    const CoordTransform& ct,
                    py::array_t<double>& xs,
                    py::array_t<double>& ys,
                    py::array_t<double>& zs
                ){
                    numpyApplyForward(ct, xs, ys, zs);
                },
                py::arg().noconvert(),
                py::arg().noconvert(),
                py::arg().noconvert()
            )
            .def(
                "applyReverse",
                [](
                    const CoordTransform& ct,
                    py::array_t<double>& xs,
                    py::array_t<double>& ys,
                    py::array_t<double>& zs
                ){
                    numpyApplyReverse(ct, xs, ys, zs);
                },
                py::arg().noconvert(),
                py::arg().noconvert(),
                py::arg().noconvert()
            )

            // RayVector versions.
            .def("applyForward", (void (CoordTransform::*)(RayVector&) const) &CoordTransform::applyForward)
            .def("applyReverse", (void (CoordTransform::*)(RayVector&) const) &CoordTransform::applyReverse)

            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const CoordTransform& ct) { return py::make_tuple(ct.getDr(), ct.getRot()); },
                [](py::tuple t) { return CoordTransform(t[0].cast<Vector3d>(), t[1].cast<Matrix3d>()); }
            ))
            .def("__hash__", [](CoordTransform& ct) {
                return py::hash(py::make_tuple(
                    "CPPCoordTransform",
                    py::tuple(py::cast(ct.getDr())),
                    py::tuple(py::cast(ct.getRot()).attr("ravel")())
                ));
            })
            .def("__repr__", &CoordTransform::repr);
    }
}
