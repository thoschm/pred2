#ifndef _COLLECTOR_H_
#define _COLLECTOR_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <Eigen/Dense>
#include <iostream>
#include <float.h>
#include "whitening.h"
#include "kmeans.h"


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{


////////////////////////////////
// time series importer
////////////////////////////////
template <typename NumericalType>
class SeriesCollector
{
    SeriesCollector(const SeriesCollector &other);
    SeriesCollector &operator=(const SeriesCollector &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    SeriesCollector(const uint seriesWindow,
                    const uint featureWindow,
                    const uint codeWords) : mSWindow(seriesWindow),
                                            mFWindow(featureWindow),
                                            mK(codeWords)
    { }

    // compute code words
    void codeWords(MatrixXt *words, const std::vector<NumericalType> &indata)
    {
        words->setZero();

        // some checks
        if (indata.size() < mSWindow)
        {
            std::cerr << "SeriesCollector: need more samples for given window size\n";
            return;
        }
        if (words->rows() != mFWindow)
        {
            std::cerr << "SeriesCollector: output matrix row count must be equal to feature window size\n";
            return;
        }
        if (words->cols() != mK)
        {
            std::cerr << "SeriesCollector: out matrix column count must be equal to the desired number of code words\n";
            return;
        }
        if (mSWindow < mFWindow)
        {
            std::cerr << "SeriesCollector: feature window should be much smaller than series window\n";
            return;
        }

        collect(words, indata);
    }


private:
    // extract features and compute code words
    void collect(MatrixXt *words, const std::vector<NumericalType> &indata)
    {
        const uint slimit = indata.size() - mSWindow, // series window
                   flimit = mSWindow - mFWindow;      // feature window

        // this will need lots of memory
        const uint cols = (slimit + 1u) * (flimit + 1u);
        MatrixXt features(mFWindow, cols);
        std::cerr << "columns allocated: " << cols << std::endl;

        // collect
        uint cnt = 0;
        for (uint i = 0; i <= slimit; ++i)
        {
            // normalze current window to 0 - 1
            NumericalType vmin, scale;
            normalize(indata, i, &vmin, &scale);
            //std::cerr << "---------" << std::endl;

            // extract feature windows from normalized series window
            for (uint k = 0; k <= flimit; ++k)
            {
                //std::cerr << "vector " << cnt << ": ";
                for (uint f = 0; f < mFWindow; ++f)
                {
                    features(f, cnt) = scale * (indata[i + k + f] - vmin);
                    //std::cerr << i + k + f << " ";
                }
                ++cnt;
                //std::cerr << std::endl;
            }
        }
        std::cerr << "vectors seen: " << cnt << std::endl;

        // begin whitening


    }

    // compute min max normalization
    void normalize(const std::vector<NumericalType> &data,
                   const uint startAt,
                   NumericalType *minVal,
                   NumericalType *scaling)
    {
        // normalize data window to 0-1
        NumericalType vmin = (NumericalType)FLT_MAX,
                      vmax = (NumericalType)-FLT_MAX;
        for (uint k = 0; k < mSWindow; ++k)
        {
            const uint idx = startAt + k;
            if (data[idx] < vmin) vmin = data[idx];
            if (data[idx] > vmax) vmax = data[idx];
        }
        NumericalType scale;
        if (vmin == vmax)
        {
            scale = (NumericalType)1.0;
            vmin -= (NumericalType)0.5;
        }
        else
        {
            scale = (NumericalType)1.0 / (vmax - vmin);
        }

        // commit
        *minVal = vmin;
        *scaling = scale;
    }

    // vars
    uint mSWindow,
         mFWindow,
         mK;
};

} // namespace

#endif
