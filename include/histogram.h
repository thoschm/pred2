#ifndef _HIST_H_
#define _HIST_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <Eigen/Dense>
#include <iostream>


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{


////////////////////////////////
// histogram generator
////////////////////////////////
template <typename NumericalType>
class RBFHistogram
{
    RBFHistogram(const RBFHistogram &other);
    RBFHistogram &operator=(const RBFHistogram &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    RBFHistogram(const uint dim, const NumericalType sigma = (NumericalType)1.0) : mDim(dim), mSigma(sigma)
    { }

    // compute histogram
    void compute(VectorXt *out, const MatrixXt &indata, const MatrixXt &words) const
    {
        out->setZero();

        // some checks
        if (indata.rows() != mDim)
        {
            std::cerr << "RBFHistogram: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (out->rows() != words.cols())
        {
            std::cerr << "RBFHistogram: output vector has invalid dimensions\n";
            return;
        }
        if (words.rows() != mDim)
        {
            std::cerr << "RBFHistogram: code words matrix has invalid dimensions\n";
            return;
        }

        const uint dsize = indata.cols(),
                   wsize = words.cols();
        // compute euclidean distance to every single code word
        for (uint i = 0; i < dsize; ++i)
        {
            // compute euclidean distance to each codeword and apply rbf
            for (uint k = 0; k < wsize; ++k)
            {
                const NumericalType dist = (indata.col(i) - words.col(k)).norm();
                (*out)[k] += std::exp(-mSigma * dist);
            }
        }
    }

private:
    uint mDim;
    NumericalType mSigma;
};

} // namespace

#endif
