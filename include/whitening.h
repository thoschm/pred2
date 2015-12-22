#ifndef _WHITENING_H_
#define _WHITENING_H_


////////////////////////////////////////
// INCLUDES
////////////////////////////////////////
#include <Eigen/Dense>


////////////////////////////////////////
// NAMESPACE
////////////////////////////////////////
namespace BOF
{


////////////////////////////////////////
// STORAGE FOR WHITENING TRANSFORM
////////////////////////////////////////
template <typename NumericalType>
struct WhiteningTransform
{
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    WhiteningTransform(const uint dataDim) : dim(dataDim), successful(false)
    {
        mean = VectorXt(dataDim);
        irot = MatrixXt(dataDim, dataDim);
        scaling = VectorXt(dataDim);
    }

    // vars
    uint dim;
    bool successful;
    VectorXt mean;
    MatrixXt irot;
    VectorXt scaling;
};


////////////////////////////////////////
// PCAWHITENING CLASS
////////////////////////////////////////
template <typename NumericalType>
class PCAWhitening
{
    PCAWhitening(const PCAWhitening &other);
    PCAWhitening &operator=(const PCAWhitening &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    PCAWhitening(const uint dim,
                 const NumericalType epsilon = (NumericalType)0.0,
                 const NumericalType smallestEVPercentage = (NumericalType)0.0) : mDim(dim),
                                                                                  mEpsilon(epsilon),
                                                                                  mEVPercentage(smallestEVPercentage)
    { }

    // compute
    void computeTransform(WhiteningTransform<NumericalType> *wt, const MatrixXt &indata)
    {
        wt->successful = false;
        // some checks
        if (indata.rows() != mDim)
        {
            std::cerr << "PCAWhitening: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (wt->dim != mDim)
        {
            std::cerr << "PCAWhitening: WhiteningTransform container has invalid dimensions\n";
            return;
        }
        whiten(wt, indata);
    }

    // compute from given transform
    void applyTransformInPlace(MatrixXt *data, const WhiteningTransform<NumericalType> &wt)
    {
        // checks
        if (data->rows() != mDim)
        {
            std::cerr << "PCAWhitening: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (wt.dim != mDim)
        {
            std::cerr << "PCAWhitening: WhiteningTransform container has invalid dimensions\n";
            return;
        }
        if (!wt.successful)
        {
            std::cerr << "PCAWhitening: WhiteningTransform is invalid\n";
            return;
        }

        // now apply transform
        const uint cols = data->cols();
        VectorXt x;
        for (uint i = 0; i < cols; ++i)
        {
            x = wt.irot * (data->col(i) - wt.mean);
            data->col(i) = (wt.scaling.array() * x.array()).matrix();
        }
    }


private:

    // whitening based in input data
    void whiten(WhiteningTransform<NumericalType> *wt, const MatrixXt &indata)
    {
        // mean and cov
        VectorXt mean(mDim);
        MatrixXt cov(mDim, mDim);
        VectorXt scaleFactors(mDim);

        // compute data mean and cov
        dataMean(&mean, indata);
        dataCov(&cov, indata, mean);

        // compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<MatrixXt> esolver;
        esolver.compute(cov);
        const VectorXt d = esolver.eigenvalues();
        const MatrixXt irot = esolver.eigenvectors().transpose();

        // get full variance
        const NumericalType fullVariance = d.sum();
        int remv = -1;
        NumericalType psum = (NumericalType)0.0;

        // mute dimensions with little variance
        for (uint i = 0; i < mDim; ++i)
        {
            const NumericalType percent = (NumericalType)100.0 * d(i) / fullVariance;
            psum += percent;
            std::cerr << percent << "% *** (" << psum << "%)";
            if (psum < mEVPercentage)
            {
                remv = i;
                std::cerr << " <-- remove\n";
            }
            else
            {
                std::cerr << "\n";
            }
        }

        // compute scaling factors
        for (uint k = 0; k < mDim; ++k)
        {
            if ((int)k <= remv)
            {
                scaleFactors(k) = (NumericalType)0.0;
                continue;
            }
            scaleFactors(k) = (NumericalType)1.0 / std::sqrt(d(k) + mEpsilon);
            if (std::isnan(scaleFactors(k)) || std::isinf(scaleFactors(k)))
            {
                std::cerr << "PCAWhitening: some dimensions are underrepresented, choose a larger epsilon value.\n";
                return;
            }
        }

        // commit
        wt->irot = irot;
        wt->mean = mean;
        wt->scaling = scaleFactors;
        wt->successful = true;
    }

    // compute mean
    void dataMean(VectorXt *out, const MatrixXt &samples) const
    {
        // no checks here, be careful with dimensions of "out" and "samples"
        out->setZero();
        const uint cols = samples.cols();
        for (uint i = 0; i < cols; ++i)
        {
            *out += samples.col(i);
        }
        *out /= cols;
    }

    // compute cov
    void dataCov(MatrixXt *out, const MatrixXt &samples, const VectorXt &mean) const
    {
        // no checks here, be careful with dimensions
        out->setZero();
        VectorXt d;
        uint cols = samples.cols();
        for (uint i = 0; i < cols; ++i)
        {
            d = samples.col(i) - mean;
            *out += d * d.transpose();
        }
        *out /= cols;
    }

    uint mDim;
    NumericalType mEpsilon,
                  mEVPercentage;
};


// namespace
}

#endif
