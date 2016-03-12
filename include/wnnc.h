#ifndef _NN_CLASSIFIER_H_
#define _NN_CLASSIFIER_H_


////////////////////////////////////////
// INCLUDE
////////////////////////////////////////
#include "KNNAdapter.h"


////////////////////////////////////////
// NAMESPACE
////////////////////////////////////////
namespace BOF
{
////////////////////////////////////////
// CLASSIFICATION RESULT
////////////////////////////////////////
template <typename NumericalType>
struct ClassificationResult
{
    uint category;
    NumericalType confidence;
    bool success;
    ClassificationResult() : category(UINT_MAX), confidence((NumericalType)0.0), success(false)
    { }
};


////////////////////////////////////////
// LABELED SAMPLE
////////////////////////////////////////
template <typename NumericalType>
struct Sample
{
    Sample(const NumericalType *array, const uint category) : data(array), label(category)
    { }
    const NumericalType *data;
    uint label;
};


////////////////////////////////////////
// SAMPLE ACCESSOR
////////////////////////////////////////
template <typename NumericalType>
struct SampleAccessor
{
private:
    SampleAccessor();
    SampleAccessor(const SampleAccessor &other);
    SampleAccessor &operator=(const SampleAccessor &other);

public:
    static NumericalType at(const Sample<NumericalType> &item, const uint i) { return item.data[i]; }
};


////////////////////////////////////////
// WEIGHTED NN CLASSIFIER
////////////////////////////////////////
template <typename NumericalType>
class WeightedNNClassifier
{
    WeightedNNClassifier();
    WeightedNNClassifier(const WeightedNNClassifier &other);
    WeightedNNClassifier &operator=(const WeightedNNClassifier &other);

public:
    typedef std::vector<Sample<NumericalType> > SampleContainer;
    typedef typename GenericKNNAdapter<-1, SampleContainer, SampleAccessor<NumericalType>, NumericalType>::IdxDistPairs IdxDistPairs;

    // ctor
    WeightedNNClassifier(const uint dim,
                         const uint categories,
                         const uint N = 17u,
                         const NumericalType gamma = (NumericalType)3.0) : mDim(dim),
                                                                           mCateg(categories),
                                                                           mN(N), mGamma(gamma),
                                                                           mContainer(NULL)
    { }

    // get data pointer
    void attach(const SampleContainer *container)
    {
        mContainer = container;
    }

    // implement classification
    ClassificationResult<NumericalType> classify(const Sample<NumericalType> &sample) const
    {
        if (!mKnn.hasTree())
        {
            std::cerr << "NNCLASSIFIER: call prepare() first.\n";
            return ClassificationResult<NumericalType>();
        }

        // get N+1 nearest neighbors
        IdxDistPairs nn = mKnn.nnSearchNSqrt(sample, mN + 1u);

        // weighted voting
        uint limit = nn.size() - 1u;
        std::vector<NumericalType> categ(mCateg, (NumericalType)0.0);
        NumericalType dist, w;
        for (uint i = 0; i < limit; ++i)
        {
            // normalize distance
            dist = nn[i].second / (nn[limit].second + (NumericalType)1e-8);
            //std::cerr << "dist: " << dist << std::endl;
            categ[mContainer->at(nn[i].first).label] += std::exp(-mGamma * dist);
        }

        // find category with biggest weight
        NumericalType best = (NumericalType)0.0;
        uint idx = UINT_MAX;
        for (uint i = 0; i < categ.size(); ++i)
        {
            std::cerr << categ[i] << " ";
            if (categ[i] > best)
            {
                best = categ[i];
                idx = i;
            }
        }
        std::cerr << std::endl;
        NumericalType secbest = (NumericalType)0.0;
        for (uint i = 0; i < categ.size(); ++i)
        {
            if (i == idx) continue;
            if (categ[i] > secbest)
            {
                secbest = categ[i];
            }
        }
        // compose result
        ClassificationResult<NumericalType> res;
        res.success = true;
        res.confidence = (NumericalType)100.0 * ((NumericalType)1.0 - secbest / best);
        res.category = idx;

        std::cerr << "best: " << best << "   secbest:" << secbest << std::endl;
        return res;
    }

    // rebuild tree
    void train()
    {
        if (mContainer == NULL || mContainer->size() == 0)
        {
            std::cerr << "NNCLASSIFIER: no training samples available.\n";
            return;
        }
        mKnn.attach(mContainer, mDim);
    }

private:
    GenericKNNAdapter<-1, SampleContainer, SampleAccessor<NumericalType>, NumericalType> mKnn;
    uint mDim, mCateg, mN;
    NumericalType mGamma;
    const SampleContainer *mContainer;
};

// namespace
}


#endif

