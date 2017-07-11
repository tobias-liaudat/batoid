#include "sphere.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<jtrace::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportSphere(py::module& m) {
        py::class_<Sphere, std::shared_ptr<Sphere>, Surface>(m, "Sphere")
            .def(py::init<double,double,double,double>(), "init",
                 "R"_a, "B"_a,
                 "Rin"_a=0.0, "Rout"_a=std::numeric_limits<double>::infinity())
            .def_property_readonly("R", &Sphere::getR)
            .def_property_readonly("B", &Sphere::getB)
            .def("sag", &Sphere::sag)
            .def("normal", &Sphere::normal)
            .def("intersect", (Intersection (Sphere::*)(const Ray&) const) &Sphere::intersect)
            .def("intersect", (std::vector<jtrace::Intersection> (Sphere::*)(const std::vector<jtrace::Ray>&) const) &Sphere::intersect)
            .def("__repr__", &Sphere::repr);
    }
}