#ifndef _RESIZE_H_
#define _RESIZE_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <iostream>
#include <cmath>
#include <float.h>


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
    LANCZOS4 = 4
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
};

}

#endif
