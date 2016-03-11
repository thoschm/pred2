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
#include "histogram.h"
#include "norm.h"
#include "dwt.h"


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
                    const uint featureSize,
                    const uint codeWords,
                    const WaveletType wavelet = D4_WAVELET) : mSWindow(seriesWindow),
                                                              mDim(featureSize),
                                                              mK(codeWords),
                                                              mWavelet(wavelet)
    { }

    // compute code words
    void codeWords(MatrixXt *words,
                   NormParams<NumericalType> *norm,
                   WhiteningTransform<NumericalType> *wt,
                   const std::vector<NumericalType> &indata) const
    {
        words->setZero();

        // some checks
        if (indata.size() < mSWindow)
        {
            std::cerr << "SeriesCollector: need more samples for given window size\n";
            return;
        }
        if (words->rows() != mDim)
        {
            std::cerr << "SeriesCollector: output matrix row count must be equal to feature dim\n";
            return;
        }
        if (words->cols() != mK)
        {
            std::cerr << "SeriesCollector: out matrix column count must be equal to the desired number of code words\n";
            return;
        }
        if (mSWindow < mDim)
        {
            std::cerr << "SeriesCollector: feature window should be much smaller than series window\n";
            return;
        }
        if (mDim < 2u)
        {
            std::cerr << "SeriesCollector: need more dimensions\n";
            return;
        }
        if (wt->dim != mDim)
        {
            std::cerr << "SeriesCollector: whitening transform has invalid dimensions\n";
            return;
        }
        if (!FastDWT<NumericalType>::isPow2(mDim))
        {
            std::cerr << "SeriesCollector: feature dimension is not a power of 2\n";
            return;
        }

        collect(words, norm, wt, indata);
    }
/*
    // get window signature
    void signature(VectorXt *sig,
                   const NormParams<NumericalType> &norm,
                   const WhiteningTransform<NumericalType> &wt,
                   const std::vector<NumericalType> &indata,
                   const MatrixXt &words,
                   const uint index = UINT_MAX) const
    {
        sig->setZero();

        // some checks
        if (indata.size() < mSWindow)
        {
            std::cerr << "SeriesCollector: need more samples for given window size\n";
            return;
        }
        if (sig->rows() != mK)
        {
            std::cerr << "SeriesCollector: signature vector must have K rows\n";
            return;
        }
        if (wt.dim != mDim)
        {
            std::cerr << "SeriesCollector: whitening transform has invalid dimensions\n";
            return;
        }
        if (!wt.successful)
        {
            std::cerr << "SeriesCollector: whitening transform is invalid\n";
            return;
        }
        if (words.rows() != mDim)
        {
            std::cerr << "SeriesCollector: code words rows number invalid\n";
            return;
        }
        if (words.cols() != mK)
        {
            std::cerr << "SeriesCollector: code words cols number invalid\n";
            return;
        }

        // TODO: use collect code

        // set window position
        const uint pos = std::min(index, (uint)(indata.size() - mSWindow)),
                   flimit = mSWindow - mDim, // feature window
                   cols = flimit + 1u;
        // alloc
        MatrixXt features(mDim, cols);

        // collect
        const NumericalType normIndex = (NumericalType)1.0 / flimit;
        // normalze current window to 0 - 1
        NumericalType vmin, scale;
        normalize(indata, pos, &vmin, &scale);

        // extract feature windows from normalized series window
        for (uint k = 0; k <= flimit; ++k)
        {
            for (uint f = 0; f < mFWindow; ++f)
            {
                features(f, k) = scale * (indata[pos + k + f] - vmin);
            }
            features(mFWindow, k) = k * normIndex;
        }
        std::cerr << features.transpose() << std::endl;

        // begin whitening
        PCAWhitening<NumericalType> pca(mDim);
        std::cout << "apply whitening..." << std::endl;
        pca.applyTransformInPlace(&features, wt);
        std::cerr << features.transpose() << std::endl;

        // gen histogram
        RBFHistogram<NumericalType> rbf(mDim, (NumericalType)1.0);
        rbf.compute(sig, features, words);
    }
    */

private:
    // extract features and compute code words
    void collect(MatrixXt *words,
                 NormParams<NumericalType> *norm,
                 WhiteningTransform<NumericalType> *wt,
                 const std::vector<NumericalType> &indata) const
    {
        const uint slimit = indata.size() - mSWindow, // series window
                   flimit = mSWindow - mDim;          // feature window

        // this will need lots of memory
        const uint cols = (slimit + 1u) * (flimit + 1u);
        MatrixXt features(mDim, cols);
        std::cerr << "columns allocated: " << cols << std::endl;

        // collect
        uint cnt = 0;
        NumericalType s[(uint)mWavelet], w[(uint)mWavelet];
        WaveletCoefficients<NumericalType>::lookup(mWavelet, s, w);
        FastDWT<NumericalType> dwt(mDim);
        for (uint i = 0; i <= slimit; ++i)
        {
            // normalze current window to 0 - 1
            NumericalType vmin, scale;
            normalize(indata, i, &vmin, &scale);

            // extract feature windows from normalized series window
            for (uint k = 0; k <= flimit; ++k)
            {
                for (uint f = 0; f < mDim; ++f)
                {
                    features(f, cnt) = scale * (indata[i + k + f] - vmin);
                }
                dwt.compute(features.col(cnt).data(), s, w, (uint)mWavelet);
                ++cnt;
            }
        }
        std::cerr << "vectors seen: " << cnt << std::endl;

        //std::cerr << features.transpose() << std::endl;

        // begin normalization
        std::cout << "compute dimension-wise normalization..." << std::endl;
        Normalization<NumericalType> elemnorm(mDim);
        elemnorm.computeParams(norm, features);
        std::cout << "apply normalization..." << std::endl;
        elemnorm.applyParamsInPlace(&features, *norm);
        //std::cerr << features.transpose() << std::endl;

        // begin whitening
        std::cerr << "compute whitening transform..." << std::endl;
        PCAWhitening<NumericalType> pca(mDim);
        pca.computeTransform(wt, features);
        std::cout << "apply whitening..." << std::endl;
        pca.applyTransformInPlace(&features, *wt);
        //std::cerr << features.transpose() << std::endl;

        // clustering
        std::cout << "clustering..." << std::endl;
        KMeans<NumericalType> kmeans(mDim, mK, 10u * mK);
        std::vector<uint> freq;
        kmeans.compute(words, &freq, features);
        for (uint i = 0; i < mK; ++i)
        {
            std::cerr << "cluster " << i << ": " << freq[i] << " supporters" << std::endl;
        }
    }

    // compute min max normalization
    void normalize(const std::vector<NumericalType> &data,
                   const uint startAt,
                   NumericalType *minVal,
                   NumericalType *scaling) const
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
         mDim,
         mK;
    WaveletType mWavelet;
};

} // namespace

#endif
