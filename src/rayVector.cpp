#include "rayVector.h"
#include "ray.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Eigen::Vector3d;

namespace batoid {
    RayVector::RayVector(
        const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
        const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
        const std::vector<double>& t, const std::vector<double>& w,
        const std::vector<double>& flux, const std::vector<bool>& vignetted,
        const CoordSys& coordSys
    ) : _coordSys(coordSys)
    {
        _rays.reserve(x.size());
        bool wSame{true};
        double w0{w[0]};
        for(int i=0; i<x.size(); i++) {
            _rays.emplace_back(x[i], y[i], z[i], vx[i], vy[i], vz[i], t[i], w[i], flux[i], vignetted[i]);
            if (w[i] != w0) wSame = false;
        }
        if (wSame) _wavelength=w0;
    }

    std::string RayVector::repr() const {
        std::ostringstream oss("RayVector([", std::ios_base::ate);
        if (_rays.size() > 0)
            oss << _rays[0];
        for(int i=1; i<_rays.size(); i++) {
            oss << ", " << _rays[i];
        }
        oss << ']';
        if (!std::isnan(_wavelength))
            oss << ", wavelength=" << _wavelength;
        oss << ')';
        return oss.str();
    }

    std::vector<double> RayVector::phase(const Vector3d& r, double t) const {
        auto result = std::vector<double>(_rays.size());
        parallelTransform(_rays.cbegin(), _rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.phase(r, t); }
        );
        return result;
    }

    std::vector<std::complex<double>> RayVector::amplitude(const Vector3d& r, double t) const {
        auto result = std::vector<std::complex<double>>(_rays.size());
        parallelTransform(_rays.cbegin(), _rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return result;
    }

    std::complex<double> RayVector::sumAmplitude(const Vector3d& r, double t) const {
        auto result = std::vector<std::complex<double>>(_rays.size());
        parallelTransform(_rays.cbegin(), _rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.amplitude(r, t); }
        );
        return std::accumulate(result.begin(), result.end(), std::complex<double>(0,0));
    }

    std::vector<Vector3d> RayVector::positionAtTime(double t) const {
        auto result = std::vector<Vector3d>(_rays.size());
        parallelTransform(_rays.cbegin(), _rays.cend(), result.begin(),
            [=](const Ray& ray)
                { return ray.positionAtTime(t); }
        );
        return result;
    }
    void RayVector::propagate(double t) {
        parallel_for_each(_rays.begin(), _rays.end(),
            [=](Ray& ray)
                { ray.propagate(t); }
        );
    }

    void RayVector::trimVignetted(double minFlux) {
        _rays.erase(
            std::remove_if(
                _rays.begin(),
                _rays.end(),
                [=](const Ray& r){ return r.failed || r.vignetted || r.flux<minFlux; }
            ),
            _rays.end()
        );
    }

    RayVector concatenateRayVectors(const std::vector<RayVector>& rvs) {
        if (rvs.size() == 0)
            return RayVector();

        int n = std::accumulate(
            rvs.begin(), rvs.end(), 0,
            [](int s, const RayVector& rv){ return s + rv.size(); }
        );
        std::vector<Ray> out;
        out.reserve(n);

        double _wavelength = rvs[0].getWavelength();
        for (const auto& rv: rvs) {
            if (_wavelength != rv.getWavelength())
                _wavelength = NAN;
            if (rv.getCoordSys() != rvs[0].getCoordSys())
                throw std::runtime_error("CoordSys's must match");
            out.insert(out.end(), rv.cbegin(), rv.cend());
        }
        return RayVector(out, rvs[0].getCoordSys(), _wavelength);
    }
}
