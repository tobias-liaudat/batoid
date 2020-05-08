#ifndef batoid_surface_h
#define batoid_surface_h

#include <vector>
#include <memory>
#include <utility>
#include "ray.h"
#include "rayVector.h"
#include "medium.h"
#include "coating.h"
#include "coordsys.h"

#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    class Surface {
    public:
        virtual ~Surface() {}

        virtual double sag(double, double) const = 0;
        virtual Vector3d normal(double, double) const = 0;
        virtual bool timeToIntersect(const Ray& r, double& t) const;

        void intersect(RayVector&, const CoordSys* cs=nullptr) const;

        void reflect(RayVector&, const Coating* coating=nullptr, const CoordSys* cs=nullptr) const;

        void refract(RayVector&, const Medium&, const Medium&, const Coating* coating=nullptr, const CoordSys* cs=nullptr) const;

        std::pair<RayVector, RayVector> rSplit(const RayVector&, const Medium&, const Medium&, const Coating&, const CoordSys* cs=nullptr) const;
        // std::pair<RayVector, RayVector> rSplitProb(const RayVector&, const Medium&, const Medium&, const Coating&, const CoordSys* cs=nullptr) const;

    private:
        // Single ray methods
        void _justIntersect(Ray&) const;

        void _justReflect(Ray&, double& alpha) const;

        void _justRefract(Ray&, const Medium&, const Medium&, double& alpha) const;
        void _justRefract(Ray&, double, double, double& alpha) const;

        std::pair<Ray, Ray> _justRSplit(const Ray&, const Medium&, const Medium&, const Coating&) const;
        std::pair<Ray, Ray> _justRSplit(const Ray&, const double, const double, const Coating&) const;
    };
}
#endif
