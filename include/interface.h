#ifndef _INTERFACE_H_
#define _INTERFACE_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <iostream>
#include <collector.h>


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{
////////////////////////////////
// PARAMETERS
////////////////////////////////
struct BOFParameters
{
    uint windowSize,
         featureSize,
         codeWords,
         numParts,
         scalingMin,
         scalingStep,
         lookAhead;
    WaveletType waveletType;
    FilterType filterType;
    float signatureSigma;

    BOFParameters() : windowSize(531u),
                      featureSize(32u),
                      codeWords(10u),
                      numParts(5u),
                      scalingMin(100u),
                      scalingStep(20u),
                      lookAhead(50u),
                      waveletType(D8_WAVELET),
                      filterType(LANCZOS8),
                      signatureSigma(1.0f)
    { }
};


////////////////////////////////
// INTERFACE
////////////////////////////////
template <typename NumericalType>
class BOFClassifier
{
    BOFClassifier(const BOFClassifier &other);
    BOFClassifier &operator=(const BOFClassifier &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    BOFClassifier(const BOFParameters &params) : mParams(params)
    { }

    void codeWords(MatrixXt *words,
                   NormParams<NumericalType> *normParams,
                   WhiteningTransform<NumericalType> *whiteningTf,
                   const std::vector<NumericalType> &indata) const
    {
        // prepare words
        words->resize(mParams.featureSize, mParams.codeWords);
        words->setZero();

        // init collector
        SeriesCollector<NumericalType> collector(mParams.windowSize,
                                                 mParams.featureSize,
                                                 mParams.codeWords,
                                                 mParams.numParts,
                                                 mParams.waveletType,
                                                 mParams.signatureSigma);
        MatrixXt features;

        // append different scaling intervals
        for (int s = 100; s >= (int)mParams.scalingMin; s -= (int)mParams.scalingStep)
        {
            std::cout << s << "% scaling:" << std::endl;
            std::vector<NumericalType> tmp;
            const uint samples = (NumericalType)0.01 * (NumericalType)s * (NumericalType)indata.size();
            Interpolator<NumericalType>::resize(&tmp, samples, indata, mParams.filterType);
            collector.append(&features, tmp);
        }

        // compute words
        collector.codeWords(words, normParams, whiteningTf, &features);
    }

    void labels(std::vector<uint> *out,
                const std::vector<NumericalType> &indata,
                uint (*const labelfunc)(const std::vector<NumericalType> &, const uint, const uint)) const
    {
        // compute cols
        uint cols = 0;
        for (int s = 100; s >= (int)mParams.scalingMin; s -= (int)mParams.scalingStep)
        {
            const uint samples = (NumericalType)0.01 * (NumericalType)s * (NumericalType)indata.size();
            const uint limit = samples - mParams.windowSize - mParams.lookAhead;
            cols += limit + 1u;
        }
        out->clear();
        out->resize(cols);
        uint cnt = 0;
        std::cerr << "computing labels..." << std::endl;
        for (int s = 100; s >= (int)mParams.scalingMin; s -= (int)mParams.scalingStep)
        {
            std::cout << s << "% scaling..." << std::endl;
            std::vector<NumericalType> tmp;
            const uint samples = (NumericalType)0.01 * (NumericalType)s * (NumericalType)indata.size();
            Interpolator<NumericalType>::resize(&tmp, samples, indata, mParams.filterType);
            const uint limit = samples - mParams.windowSize - mParams.lookAhead;

            // compute labels
            for (uint i = 0; i <= limit; ++i)
            {
                const uint last = i + mParams.windowSize - 1u;
                out->at(cnt++) = labelfunc(tmp, last, last + mParams.lookAhead);
            }
        }
    }

    void signatures(MatrixXt *features,
                    const MatrixXt &words,
                    const NormParams<NumericalType> &normParams,
                    const WhiteningTransform<NumericalType> &whiteningTf,
                    const std::vector<NumericalType> &indata) const
    {
        // init collector
        SeriesCollector<NumericalType> collector(mParams.windowSize,
                                                 mParams.featureSize,
                                                 mParams.codeWords,
                                                 mParams.numParts,
                                                 mParams.waveletType,
                                                 mParams.signatureSigma);

        // create signatures
        const uint sigsize = mParams.codeWords * mParams.numParts;

        // compute cols
        uint cols = 0;
        for (int s = 100; s >= (int)mParams.scalingMin; s -= (int)mParams.scalingStep)
        {
            const uint samples = (NumericalType)0.01 * (NumericalType)s * (NumericalType)indata.size();
            const uint limit = samples - mParams.windowSize - mParams.lookAhead;
            cols += limit + 1u;
        }

        // prepare
        features->resize(sigsize, cols);
        features->setZero();

        // loop different scaling intervals
        VectorXt onesig(sigsize);
        uint cnt = 0;
        std::cerr << "computing signatures..." << std::endl;
        for (int s = 100; s >= (int)mParams.scalingMin; s -= (int)mParams.scalingStep)
        {
            std::cout << s << "% scaling..." << std::endl;
            std::vector<NumericalType> tmp;
            const uint samples = (NumericalType)0.01 * (NumericalType)s * (NumericalType)indata.size();
            Interpolator<NumericalType>::resize(&tmp, samples, indata, mParams.filterType);
            const uint limit = samples - mParams.windowSize - mParams.lookAhead;

            // compute sigs
            for (uint i = 0; i <= limit; ++i)
            {
                collector.signature(&onesig, normParams, whiteningTf, tmp, words, i);
                features->col(cnt++) = onesig;
            }
        }
        //std::cerr << "cols " << cols << "   cnt " << cnt << std::endl;
    }

    void signature(MatrixXt *sig,
                   const MatrixXt &words,
                   const NormParams<NumericalType> &normParams,
                   const WhiteningTransform<NumericalType> &whiteningTf,
                   const std::vector<NumericalType> &indata,
                   const uint index = UINT_MAX) const
    {
        const uint sigsize = mParams.codeWords * mParams.numParts;
        sig->resize(sigsize, 1u);
        sig->setZero();

        // init collector
        VectorXt tmp(sigsize);
        SeriesCollector<NumericalType> collector(mParams.windowSize,
                                                 mParams.featureSize,
                                                 mParams.codeWords,
                                                 mParams.numParts,
                                                 mParams.waveletType,
                                                 mParams.signatureSigma);
        collector.signature(&tmp, normParams, whiteningTf, indata, words, index);
        sig->col(0) = tmp;
    }

    void computeNormWhite(MatrixXt *data, NormParams<NumericalType> *norm, WhiteningTransform<NumericalType> *wtf, const uint dim)
    {
        Normalization<NumericalType> elemnorm(dim);
        std::cerr << "compute normalization params..." << std::endl;
        elemnorm.computeParams(norm, *data);
        std::cout << "apply normalization..." << std::endl;
        elemnorm.applyParamsInPlace(data, *norm);
        std::cerr << "compute whitening transform..." << std::endl;
        PCAWhitening<NumericalType> pca(dim);
        pca.computeTransform(wtf, *data);
        std::cout << "apply whitening..." << std::endl;
        pca.applyTransformInPlace(data, *wtf);
    }

    void forwardNormWhite(MatrixXt *data, const NormParams<NumericalType> &norm, const WhiteningTransform<NumericalType> &wtf, const uint dim)
    {
        Normalization<NumericalType> elemnorm(dim);
        elemnorm.applyParamsInPlace(data, norm);
        PCAWhitening<NumericalType> pca(dim);
        pca.applyTransformInPlace(data, wtf);
    }

    void inverseNormWhite(MatrixXt *data, const NormParams<NumericalType> &norm, const WhiteningTransform<NumericalType> &wtf, const uint dim)
    {
        std::cerr << "inverse whitening and inverse normalization..." << std::endl;
        PCAWhitening<float> pca(dim);
        pca.inverseTransformInPlace(data, wtf);
        Normalization<float> elemnorm(dim);
        elemnorm.inverseParamsInPlace(data, norm);
    }

private:
    BOFParameters mParams;
};

}

#endif

