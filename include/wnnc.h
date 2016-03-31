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
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    static NumericalType at(const VectorXt &item, const uint i) { return item(i); }
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
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;
    typedef typename GenericKNNAdapter<-1, MatrixXt, VectorXt, SampleAccessor<NumericalType>, NumericalType>::IdxDistPairs IdxDistPairs;

    // ctor
    WeightedNNClassifier(const uint dim,
                         const uint categories,
                         const uint N = 17u,
                         const NumericalType gamma = (NumericalType)3.0) : mDim(dim),
                                                                           mCateg(categories),
                                                                           mN(N), mGamma(gamma),
                                                                           mContainer(NULL),
                                                                           mLabels(NULL)
    { }

    // get data pointer
    void attach(const MatrixXt *container, const std::vector<uint> *labels)
    {
        if (container->rows() != mDim)
        {
            std::cerr << "WNNC: container rows must be equal to feature dim.\n";
            return;
        }
        if (container->cols() != labels->size())
        {
            std::cerr << "WNNC: invalid number of labels.\n";
            return;
        }
        mContainer = container;
        mLabels = labels;
    }

    // implement classification
    ClassificationResult<NumericalType> classify(const VectorXt &sample) const
    {
        if (!mKnn.hasTree())
        {
            std::cerr << "WNNC: call train() first.\n";
            return ClassificationResult<NumericalType>();
        }

        // get N+1 nearest neighbors
        IdxDistPairs nn = mKnn.nnSearchNSqrt(sample, mN + 1u);

        // weighted voting
        uint limit = nn.size() - 1u;
        std::vector<NumericalType> categ(mCateg, (NumericalType)0.0);
        NumericalType dist;
        for (uint i = 0; i < limit; ++i)
        {
            // normalize distance
            dist = nn[i].second / (nn[limit].second + (NumericalType)1e-8);
            //std::cerr << "dist: " << dist << std::endl;
            categ[mLabels->at(nn[i].first)] += std::exp(-mGamma * dist);
        }

        // find category with biggest weight
        NumericalType best = (NumericalType)0.0;
        uint idx = UINT_MAX;
        for (uint i = 0; i < categ.size(); ++i)
        {
            //std::cerr << categ[i] << " ";
            if (categ[i] > best)
            {
                best = categ[i];
                idx = i;
            }
        }
        //std::cerr << std::endl;
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

        //std::cerr << "best: " << best << "   secbest:" << secbest << std::endl;
        return res;
    }

    // rebuild tree
    void train()
    {
        if (mContainer == NULL || mContainer->size() == 0)
        {
            std::cerr << "WNNC: no training samples available.\n";
            return;
        }
        mKnn.attach(mContainer, mDim);
    }

    // dump feature space into file
    void dump(const std::string &file) const
    {
        std::ofstream of;
        of.open(file.c_str(), std::ios::out);
        of << "# dim1, dim2, dim3, ..., dimN, classID\n";
        for (uint i = 0; i < mContainer->cols(); ++i)
        {
            for (uint k = 0; k < mDim; ++k)
            {
                of << (*mContainer)(k, i) << " ";
            }
            of << mLabels->at(i) << "\n";
        }
        of.close();
    }

private:
    GenericKNNAdapter<-1, MatrixXt, VectorXt, SampleAccessor<NumericalType>, NumericalType> mKnn;
    uint mDim, mCateg, mN;
    NumericalType mGamma;
    const MatrixXt *mContainer;
    const std::vector<uint> *mLabels;
};

// namespace
}


#endif

