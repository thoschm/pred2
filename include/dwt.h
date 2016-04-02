#ifndef __DWT_H__
#define __DWT_H__


////////////////////////////////////
// Includes
////////////////////////////////////
#include <iostream>
#include <string.h>


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{


////////////////////////////////////
// Supported Wavelets
////////////////////////////////////
enum WaveletType
{
    NO_WAVELET = 1, // DISABLE DWT
    D2_WAVELET = 2, // HAAR
    D4_WAVELET = 4,
    D6_WAVELET = 6,
    D8_WAVELET = 8
};


////////////////////////////////////
// Look up Wavelet Coefficients
////////////////////////////////////
template <typename NumericalType>
class WaveletCoefficients
{
    WaveletCoefficients();
    WaveletCoefficients(const WaveletCoefficients &other);
    WaveletCoefficients &operator=(const WaveletCoefficients &other);

public:
    static bool lookup(const WaveletType type,
                       NumericalType *scalingfilter,
                       NumericalType *waveletfilter)
    {
        if (type == D2_WAVELET)
        {
            scalingfilter[0] = (NumericalType)7.071067811865475244008443621048490392848359376884740365883398e-01;
            scalingfilter[1] = (NumericalType)7.071067811865475244008443621048490392848359376884740365883398e-01;
            waveletfilter[0] =  scalingfilter[1];
            waveletfilter[1] = -scalingfilter[0];
        }
        else if (type == D4_WAVELET)
        {
            scalingfilter[0] = (NumericalType)4.829629131445341433748715998644486838169524195042022752011715e-01;
            scalingfilter[1] = (NumericalType)8.365163037378079055752937809168732034593703883484392934953414e-01;
            scalingfilter[2] = (NumericalType)2.241438680420133810259727622404003554678835181842717613871683e-01;
            scalingfilter[3] = (NumericalType)-1.294095225512603811744494188120241641745344506599652569070016e-01;
            waveletfilter[0] =  scalingfilter[3];
            waveletfilter[1] = -scalingfilter[2];
            waveletfilter[2] =  scalingfilter[1];
            waveletfilter[3] = -scalingfilter[0];
        }
        else if (type == D6_WAVELET)
        {
            scalingfilter[0] = (NumericalType)3.326705529500826159985115891390056300129233992450683597084705e-01;
            scalingfilter[1] = (NumericalType)8.068915093110925764944936040887134905192973949948236181650920e-01;
            scalingfilter[2] = (NumericalType)4.598775021184915700951519421476167208081101774314923066433867e-01;
            scalingfilter[3] = (NumericalType)-1.350110200102545886963899066993744805622198452237811919756862e-01;
            scalingfilter[4] = (NumericalType)-8.544127388202666169281916918177331153619763898808662976351748e-02;
            scalingfilter[5] = (NumericalType)3.522629188570953660274066471551002932775838791743161039893406e-02;
            waveletfilter[0] =  scalingfilter[5];
            waveletfilter[1] = -scalingfilter[4];
            waveletfilter[2] =  scalingfilter[3];
            waveletfilter[3] = -scalingfilter[2];
            waveletfilter[4] =  scalingfilter[1];
            waveletfilter[5] = -scalingfilter[0];
        }
        else if (type == D8_WAVELET)
        {
            scalingfilter[0] = (NumericalType)2.303778133088965008632911830440708500016152482483092977910968e-01;
            scalingfilter[1] = (NumericalType)7.148465705529156470899219552739926037076084010993081758450110e-01;
            scalingfilter[2] = (NumericalType)6.308807679298589078817163383006152202032229226771951174057473e-01;
            scalingfilter[3] = (NumericalType)-2.798376941685985421141374718007538541198732022449175284003358e-02;
            scalingfilter[4] = (NumericalType)-1.870348117190930840795706727890814195845441743745800912057770e-01;
            scalingfilter[5] = (NumericalType)3.084138183556076362721936253495905017031482172003403341821219e-02;
            scalingfilter[6] = (NumericalType)3.288301166688519973540751354924438866454194113754971259727278e-02;
            scalingfilter[7] = (NumericalType)-1.059740178506903210488320852402722918109996490637641983484974e-02;
            waveletfilter[0] =  scalingfilter[7];
            waveletfilter[1] = -scalingfilter[6];
            waveletfilter[2] =  scalingfilter[5];
            waveletfilter[3] = -scalingfilter[4];
            waveletfilter[4] =  scalingfilter[3];
            waveletfilter[5] = -scalingfilter[2];
            waveletfilter[6] =  scalingfilter[1];
            waveletfilter[7] = -scalingfilter[0];
        }
        else if (type == NO_WAVELET)
        {
            return true;
        }
        else
        {
            std::cerr << "wavelet type not supported." << std::endl;
            return false;
        }
        return true;
    }
};


////////////////////////////////////
// Fast Discrete Wavelet Transform
////////////////////////////////////
template <typename NumericalType>
class FastDWT
{
    FastDWT(const FastDWT &other);
    FastDWT &operator=(const FastDWT &other);

public:
    FastDWT(const uint windowSize) : mWindow(windowSize)
    {
        // check size
        if (!isPow2(windowSize))
        {
            std::cerr << "FastDWT: input value count is not a power of 2." << std::endl;
        }
        mTmp = (NumericalType *)malloc(windowSize * sizeof(NumericalType));
    }

    ~FastDWT()
    {
        free(mTmp);
    }

    void setZero(NumericalType *input,
                 const uint start,
                 const uint end)
    {
        for (uint i = start; i <= end; ++i)
        {
            input[i] = (NumericalType)0.0;
        }
    }

    NumericalType energy(const NumericalType *input)
    {
        NumericalType e = (NumericalType)0.0;
        for (uint i = 0; i < mWindow; ++i)
        {
            e += input[i] * input[i];
        }
        return e;
    }

    bool compute(NumericalType *input,
                 const NumericalType *scalingfilter,
                 const NumericalType *waveletfilter,
                 const uint fcnt)
    {
        if (fcnt <= 1u) return true;

        // forward fdwt
        for (uint window = mWindow; window >= 2u; window >>= 1)
        {
            const uint half = window >> 1;
            const uint limit = window - 1u;
            for (uint i = 0, k = 0; i < limit; i += 2u, ++k)
            {
                mTmp[k]        = (NumericalType)0.0;
                mTmp[k + half] = (NumericalType)0.0;
                for (uint f = 0; f < fcnt; ++f)
                {
                    const uint curidx = (i + f) & limit;
                    mTmp[k]        += input[curidx] * scalingfilter[f];
                    mTmp[k + half] += input[curidx] * waveletfilter[f];
                }
            }
            // copy back
            memcpy(input, mTmp, window * sizeof(NumericalType));
        }
        return true;
    }

    bool inverse(NumericalType *input,
                 const NumericalType *scalingfilter,
                 const NumericalType *waveletfilter,
                 const uint fcnt)
    {
        if (fcnt <= 1u) return true;

        // inverse fdwt
        const uint fwindow = fcnt - 1u;
        for (uint window = 2u; window <= mWindow; window <<= 1)
        {
            const uint half = window >> 1;
            const uint limit = window - 1u;
            memset(mTmp, 0, window * sizeof(NumericalType));
            for (uint i = 0, k = 0; i < limit; i += 2u, ++k)
            {
                for (uint f = 0; f < fwindow; f += 2u)
                {
                    const uint idx = (i + f) & limit,
                               kh  = k + half;
                    mTmp[idx]      += input[k] * scalingfilter[f] + input[kh] * waveletfilter[f];
                    mTmp[idx + 1u] += input[k] * scalingfilter[f + 1u] + input[kh] * waveletfilter[f + 1u];
                }
            }
            // copy back
            memcpy(input, mTmp, window * sizeof(NumericalType));
        }
        return true;
    }

    static bool isPow2(const uint num)
    {
        return ((num != 0) && !(num & (num - 1u)));
    }

private:
    uint mWindow;
    NumericalType *mTmp;
};

} // namespace

#endif
