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

    BOFParameters() : windowSize(515u),
                      featureSize(16u),
                      codeWords(10u),
                      numParts(4u),
                      scalingMin(20u),
                      scalingStep(10u),
                      lookAhead(10u),
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
    typedef std::vector<Sample<NumericalType> > SampleVector;

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

    void signatures(SampleVector *out,
                    const MatrixXt &words,
                    const NormParams<NumericalType> &normParams,
                    const WhiteningTransform<NumericalType> &whiteningTf,
                    const std::vector<NumericalType> &indata) const
    {
        // prepare
        out->clear();

        // init collector
        SeriesCollector<NumericalType> collector(mParams.windowSize,
                                                 mParams.featureSize,
                                                 mParams.codeWords,
                                                 mParams.numParts,
                                                 mParams.waveletType,
                                                 mParams.signatureSigma);

        // create signatures
        const uint limit = indata.size() - mParams.windowSize - mParams.lookAhead;
        const uint sigsize = mParams.codeWords * mParams.numParts;
        for (uint i = 0; i <= limit; ++i)
        {
            out->push_back(Sample<NumericalType>(sigsize));
            collector.signature(&(out->back().signature), normParams, whiteningTf, indata, words, i);
        }
    }

private:
    BOFParameters mParams;
};

}

#endif

