#include "surface.h"
#include "coordtransform.h"
#include "utils.h"

namespace batoid {

    //
    // Single Ray "atomic" methods.
    //

    void Surface::_justIntersect(Ray& r) const {
        if (r.failed) return;
        double t;
        if (!timeToIntersect(r, t)) {
            r.failed=true;
            r.vignetted=true;
            return;
        }
        r.propagate(t);
    }

    void Surface::_justReflect(Ray& r, double& alpha) const {
        // assume we've already intersected.
        if (r.failed) return;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        alpha = r.v.dot(normVec);
        r.v -= 2*alpha*normVec;
    }

    void Surface::_justRefract(Ray& r, const Medium& m1, const Medium& m2, double& alpha) const {
        if (r.failed) return;
        _justRefract(r, m1.getN(r.wavelength), m2.getN(r.wavelength), alpha);
    }

    void Surface::_justRefract(Ray& r, double n1, double n2, double& alpha) const {
        if (r.failed) return;
        Vector3d i = r.v * n1;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        alpha = i.dot(normVec);
        if (alpha > 0.) {
            normVec *= -1;
            alpha *= -1;
        }
        double eta = n1/n2;
        double sinsqr = eta*eta*(1-alpha*alpha);
        Vector3d t = eta * i - (eta * alpha + std::sqrt(1 - sinsqr)) * normVec;
        r.v = t/n2;
    }

    std::pair<Ray, Ray> Surface::_justRSplit(const Ray& r, double n1, double n2, const Coating& coating) const {
        if (r.failed) return std::make_pair(r, r);
        // Common calculations
        Vector3d i = r.v * n1;
        Vector3d normVec(normal(r.r[0], r.r[1]));
        double alpha = i.dot(normVec);
        if (alpha > 0.) {
            normVec *= -1.;
            alpha *= -1;
        }

        // Flux coefficients
        double reflect, transmit;
        coating.getCoefs(r.wavelength, alpha, reflect, transmit);

        // Reflection calculation
        Ray reflectedRay(
            r.r, (i - 2*alpha*normVec)/n1,
            r.t, r.wavelength, reflect*r.flux, r.vignetted
        );

        // Refraction calculation
        Ray refractedRay(r);
        double eta = n1/n2;
        double sinsqr = eta*eta*(1-alpha*alpha);
        Vector3d t = eta * i - (eta * alpha + std::sqrt(1 - sinsqr)) * normVec;
        refractedRay.v = t/n2;
        refractedRay.flux *= transmit;
        return std::make_pair(reflectedRay, refractedRay);
    }

    std::pair<Ray, Ray> Surface::_justRSplit(const Ray& r, const Medium& m1, const Medium& m2, const Coating& coating) const {
        double n1 = m1.getN(r.wavelength);
        double n2 = m2.getN(r.wavelength);
        return _justRSplit(r, n1, n2, coating);
    }

    //
    // RayVector methods
    //

    void Surface::intersect(RayVector& rv, const CoordSys* cs) const {
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform ct(rv.getCoordSys(), *cs);
        parallel_for_each(
            rv.begin(), rv.end(),
            [this, ct](Ray& r) {
                ct.applyForward(r);
                _justIntersect(r);
            }
        );
        rv.setCoordSys(*cs);
    }

    void Surface::reflect(RayVector& rv, const Coating* coating, const CoordSys* cs) const {
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform ct(rv.getCoordSys(), *cs);
        parallel_for_each(
            rv.begin(), rv.end(),
            [this,ct,coating](Ray& r) {
                ct.applyForward(r);
                _justIntersect(r);
                double alpha;
                _justReflect(r, alpha);
                if (coating)
                    r.flux *= coating->getReflect(r.wavelength, alpha/r.v.norm());
            }
        );
        rv.setCoordSys(*cs);
    }

    void Surface::refract(RayVector& rv, const Medium& m1, const Medium& m2, const Coating* coating, const CoordSys* cs) const {
        if (!cs)
            cs = &rv.getCoordSys();
        CoordTransform ct(rv.getCoordSys(), *cs);
        if (std::isnan(rv.getWavelength())) {
            parallel_for_each(
                rv.begin(), rv.end(),
                [this,&m1,&m2,coating,ct](Ray& r) {
                    double alpha;
                    ct.applyForward(r);
                    _justIntersect(r);
                    _justRefract(r, m1, m2, alpha);
                    if (coating)
                        r.flux *= coating->getTransmit(r.wavelength, alpha/r.v.norm());
                }
            );
        } else {
            double n1 = m1.getN(rv.getWavelength());
            double n2 = m2.getN(rv.getWavelength());
            parallel_for_each(
                rv.begin(), rv.end(),
                [this,n1,n2,coating,ct](Ray& r) {
                    double alpha;
                    ct.applyForward(r);
                    _justIntersect(r);
                    _justRefract(r, n1, n2, alpha);
                    if (coating)
                        r.flux *= coating->getTransmit(r.wavelength, alpha/r.v.norm());
                }
            );
        }
        rv.setCoordSys(*cs);
    }

    std::pair<RayVector, RayVector> Surface::rSplit(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating, const CoordSys* cs) const {
        RayVector reflected(rv);
        RayVector refracted(rv);

        reflect(reflected, &coating, cs);
        refract(refracted, m1, m2, &coating, cs);

        return std::make_pair(reflected, refracted);
    }

    // std::pair<RayVector, RayVector> Surface::rSplitProb(const RayVector& rv, const Medium& m1, const Medium& m2, const Coating& coating, const CoordTransform* ct) const {
    //     RayVector reflected(rv);
    //     RayVector refracted(rv);
    //
    //     reflectInPlace(reflected, nullptr, ct);
    //     refractInPlace(refracted, m1, m2, nullptr, ct);
    //
    //     // Go through and probabilistically accept/reject each ray?
    //     double reflect, transmit, alpha, ran;
    //     for(unsigned int i=0; i<rv.size(); i++) {
    //         // Need to recompute the normal vector and alpha=cos(theta)...  for the third time...
    //         Vector3d normVec(normal(rv[i].r[0], rv[i].r[1]));
    //         alpha = rv[i].v.normalized().dot(normVec);
    //         coating.getCoefs(rv[i].wavelength, alpha, reflect, transmit);
    //         ran = std::uniform_real_distribution<>(0.0, 1.0)(rng);
    //         if (ran < reflect) { //choose reflect
    //             refracted[i].vignetted=true;
    //         } else if (ran < reflect+transmit) { // choose refract
    //             reflected[i].vignetted=true;
    //         } else { // choose neither
    //             refracted[i].vignetted=true;
    //             reflected[i].vignetted=true;
    //         }
    //     }
    //     reflected.trimVignettedInPlace(0.0);
    //     refracted.trimVignettedInPlace(0.0);
    //     return std::make_pair(reflected, refracted);
    // }

    bool Surface::timeToIntersect(const Ray& r, double& t) const {
        // Note t should be a good guess coming in for stability.
        // Algorithm is:
        // x,y,z <- ray.position(t)
        // sag <- surface.sag(x,y)
        // if z == sag
        //   return
        // normVec <- surface.normal(x,y)
        // plane <- Plane((x,y,sag), normVec)
        // t <- plane.intersect(r)
        // x,z,y <- ray.position(t)
        // sag <- surface.sag(x,y)
        // if z == sag
        //   return
        // ...
        Vector3d rayPoint = r.positionAtTime(t);
        double surfaceZ = sag(rayPoint[0], rayPoint[1]);
        size_t iter=0;
        double err = std::abs(surfaceZ - rayPoint[2]);
        while (err > 1e-12 && iter < 50) {
            Vector3d normVec = normal(rayPoint[0], rayPoint[1]);
            Vector3d surfacePoint{rayPoint[0], rayPoint[1], surfaceZ};
            t = normVec.dot(surfacePoint - r.r) / normVec.dot(r.v) + r.t;
            rayPoint = r.positionAtTime(t);
            surfaceZ = sag(rayPoint[0], rayPoint[1]);
            iter++;
            err = std::abs(surfaceZ - rayPoint[2]);
        }
        if (iter == 50)
            return false;
        return true;
    }

}
