#include "coordtransform.h"
#include "utils.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace batoid {

    // x is global coordinate
    // y is destination, with corresponding R and dr
    // z is source, with corresponding S and ds
    //
    // y = Rinv(x-dr)
    // z = Sinv(x-ds)
    // implies
    // x = S z + ds
    //
    // y = Rinv(S z + ds - dr)
    //   = Rinv S z + Rinv ds - Rinv dr
    //   = Rinv S z + Rinv S Sinv ds - Rinv S Sinv dr
    //   = Rinv S (z + Sinv ds - Sinv dr)
    //   = (Sinv R)^-1 (z - (Sinv dr - Sinv ds))
    //   = (Sinv R)^-1 (z - Sinv (dr - ds))

    CoordTransform::CoordTransform(const CoordSys& source, const CoordSys& destination) :
        _dr(source.m_rot.transpose()*(destination.m_origin - source.m_origin)),
        _rot(source.m_rot.transpose()*destination.m_rot),
        _source(source), _destination(destination) {}

    CoordTransform::CoordTransform(const Vector3d& dr, const Matrix3d& rot) :
        _dr(dr), _rot(rot) {}

    CoordTransform::CoordTransform() : _dr(Vector3d::Zero()), _rot(Matrix3d::Identity()) {}

    // We actively shift and rotate the coordinate system axes,
    // This looks like y = R x + dr
    // For a passive transformation of a fixed vector from one coord sys to another
    // though, we want the opposite transformation: y = R^-1 (x - dr)
    void CoordTransform::applyForward(DRef<Vector3d> r) const {
        r = _rot.transpose()*(r-_dr);
    }

    void CoordTransform::applyReverse(DRef<Vector3d> r) const {
        r = _rot*r+_dr;
    }

    void CoordTransform::applyForward(Ray& r) const {
        if (r.failed) return;
        r.r = _rot.transpose()*(r.r-_dr);
        r.v = _rot.transpose()*r.v;
    }

    void CoordTransform::applyReverse(Ray& r) const {
        if (r.failed) return;
        r.r = _rot*r.r+_dr;
        r.v = _rot*r.v;
    }

    void CoordTransform::applyForward(RayVector& rv) const {
        // assert rv.coordSys == getSource();
        parallel_for_each(rv.begin(), rv.end(),
            [this](Ray& r) { applyForward(r); }
        );
        rv.setCoordSys(CoordSys(getDestination()));
    }

    void CoordTransform::applyReverse(RayVector& rv) const {
        // assert rv.coordSys == getDestination();
        parallel_for_each(rv.begin(), rv.end(),
            [this](Ray& r) { applyReverse(r); }
        );
        rv.setCoordSys(CoordSys(getSource()));
    }

    bool operator==(const CoordTransform& ct1, const CoordTransform& ct2) {
        return ct1.getRot() == ct2.getRot() &&
               ct1.getDr() == ct2.getDr();
    }

    bool operator!=(const CoordTransform& ct1, const CoordTransform& ct2) {
        return !(ct1 == ct2);
    }

    std::ostream& operator<<(std::ostream &os, const CoordTransform& ct) {
        return os << ct.repr();
    }

}
