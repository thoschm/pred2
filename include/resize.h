#ifndef _RESIZE_H_
#define _RESIZE_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <iostream>
#include <cmath>
#include <float.h>
#include <vector>


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{
////////////////////////////////
// FILTER TYPE
////////////////////////////////
enum FilterType
{
    LANCZOS2 = 2,
    LANCZOS3 = 3,
    LANCZOS4 = 4,
    LANCZOS5 = 5,
    LANCZOS6 = 6,
    LANCZOS7 = 7,
    LANCZOS8 = 8
};


////////////////////////////////
// INTERPOLATION
////////////////////////////////
template <typename NumericalType>
class Interpolator
{
    Interpolator();
    Interpolator(const Interpolator &other);
    Interpolator &operator=(const Interpolator &other);

public:
    static NumericalType lanczos(const FilterType type, const NumericalType x)
    {
        const NumericalType fmax = (NumericalType)type;
        if (x < -fmax || x > fmax)
        {
            return (NumericalType)0.0;
        }
        else if (x > -FLT_EPSILON && x < FLT_EPSILON)
        {
            return (NumericalType)1.0;
        }
        const NumericalType xpi = x * (NumericalType)M_PI;
        const NumericalType xpidivlobes = xpi / fmax;
        return std::sin(xpi) * std::sin(xpidivlobes) / (xpi * xpidivlobes);
    }

    static void dump(const std::string &file,
                     const FilterType type,
                     const NumericalType from,
                     const NumericalType to,
                     const NumericalType step)
    {
        std::ofstream ofs;
        ofs.open(file.c_str(), std::ios::out);
        for (NumericalType f = from; f <= to; f += step)
        {
            ofs << f << " " << lanczos(type, f) << std::endl;
        }
        ofs.close();
    }

    static void dump(const std::string &file,
                     const std::vector<NumericalType> &samples,
                     const FilterType type,
                     const NumericalType step)
    {
        std::ofstream ofs;
        ofs.open(file.c_str(), std::ios::out);
        const NumericalType limit = (NumericalType)(samples.size() - 1u);
        for (NumericalType f = (NumericalType)0.0; f <= limit; f += step)
        {
            ofs << f << " " << peek(samples, type, f) << std::endl;
        }
        ofs.close();
    }

    static NumericalType peek(const std::vector<NumericalType> &samples,
                              const FilterType type,
                              const NumericalType x)
    {
        const long lobes  = (long)type,
                   xflr   = (long)std::floor(x),
                   idxmin = std::max(xflr - lobes + 1l, (long)0),
                   idxmax = std::min(xflr + lobes, (long)samples.size() - 1l);
        NumericalType sum = (NumericalType)0.0;
        for (long idx = idxmin; idx <= idxmax; ++idx)
        {
            sum += samples[idx] * lanczos(type, x - (NumericalType)idx);
        }
        return sum;
    }

    static void boxsmooth(std::vector<NumericalType> *out,
                          const std::vector<NumericalType> &samples,
                          const uint boxsize)
    {
        out->clear();
        out->resize(samples.size() - boxsize + 1u);
        for (uint i = 0; i < out->size(); ++i)
        {
            NumericalType sum = (NumericalType)0.0;
            for (uint k = 0; k < boxsize; ++k)
            {
                sum += samples[i + k];
            }
            out->at(i) = sum / boxsize;
        }
    }

    static void resize(std::vector<NumericalType> *out,
                       const uint outSamples,
                       const std::vector<NumericalType> &samples,
                       const FilterType type,
                       const uint boxsize)
    {
        std::vector<NumericalType> tmp;
        tmp.resize(outSamples + boxsize - 1u);
        const NumericalType step = (NumericalType)samples.size() / (NumericalType)tmp.size();
        for (uint i = 0; i < tmp.size(); ++i)
        {
            tmp[i] = peek(samples, type, step * (NumericalType)i);
        }
        boxsmooth(out, tmp, boxsize);
    }
};

}

#endif
