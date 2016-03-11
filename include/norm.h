#ifndef _NORM_H_
#define _NORM_H_


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
// STORAGE FOR NORM PARAMS
////////////////////////////////////////
template <typename NumericalType>
struct NormParams
{
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    NormParams(const uint dataDim) : dim(dataDim), successful(false)
    {
        mean = VectorXt(dataDim);
        stddev = VectorXt(dataDim);
    }

    // vars
    uint dim;
    bool successful;
    VectorXt mean,
             stddev;
};


////////////////////////////////////////
// NORMALIZATION CLASS
////////////////////////////////////////
template <typename NumericalType>
class Normalization
{
    Normalization(const Normalization &other);
    Normalization &operator=(const Normalization &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    Normalization(const uint dim) : mDim(dim)
    { }

    // compute
    void computeParams(NormParams<NumericalType> *params, const MatrixXt &indata) const
    {
        params->successful = false;
        // some checks
        if (indata.rows() != mDim)
        {
            std::cerr << "Normalization: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (params->dim != mDim)
        {
            std::cerr << "Normalization: NormParams container has invalid dimensions\n";
            return;
        }

        // compute
        dataMean(&(params->mean), indata);
        stdDev(&(params->stddev), indata, params->mean);

        // done
        params->successful = true;
        return;
    }

    // compute from given transform
    void applyParamsInPlace(MatrixXt *data, const NormParams<NumericalType> &params) const
    {
        // checks
        if (data->rows() != mDim)
        {
            std::cerr << "Normalization: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (params.dim != mDim)
        {
            std::cerr << "Normalization: NormParams container has invalid dimensions\n";
            return;
        }
        if (!params.successful)
        {
            std::cerr << "Normalization: NormParams are invalid\n";
            return;
        }

        // now apply transform
        const uint cols = data->cols();
        VectorXt x;
        for (uint i = 0; i < cols; ++i)
        {
            x = data->col(i) - params.mean;
            data->col(i) = (x.array() / params.stddev.array()).matrix();
        }
    }

    // compute inverse
    void inverseParamsInPlace(MatrixXt *data, const NormParams<NumericalType> &params) const
    {
        // checks
        if (data->rows() != mDim)
        {
            std::cerr << "Normalization: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (params.dim != mDim)
        {
            std::cerr << "Normalization: NormParams container has invalid dimensions\n";
            return;
        }
        if (!params.successful)
        {
            std::cerr << "Normalization: NormParams are invalid\n";
            return;
        }

        // now apply transform
        const uint cols = data->cols();
        VectorXt x;
        for (uint i = 0; i < cols; ++i)
        {
            x = (data->col(i).array() * params.stddev.array()).matrix();
            data->col(i) = x + params.mean;
        }
    }

private:
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

    // compute dimension-wise stddev
    void stdDev(VectorXt *out, const MatrixXt &samples, const VectorXt &mean) const
    {
        // no checks here, be careful with dimensions
        out->setZero();
        uint cols = samples.cols();
        for (uint d = 0; d < mDim; ++d)
        {
            NumericalType sum = (NumericalType)0.0;
            for (uint i = 0; i < cols; ++i)
            {
                const NumericalType diff = samples(d, i) - mean(d);
                sum += diff * diff;
            }
            (*out)(d) = std::sqrt(sum / cols);
            if ((*out)(d) == (NumericalType)0.0)
            {
                std::cerr << "sdtDev: dimension has zero standard deviation\n";
                (*out)(d) = (NumericalType)1.0;
            }
        }
    }

    uint mDim;
};


// namespace
}

#endif
