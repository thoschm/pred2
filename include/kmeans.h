
#ifndef _KMEANS_H_
#define _KMEANS_H_


////////////////////////////////
// INCLUDES
////////////////////////////////
#include <Eigen/Dense>
#include <iostream>
#include <float.h>
#include <vector>
#include "XorShift.h"


////////////////////////////////
// NAMESPACE
////////////////////////////////
namespace BOF
{


////////////////////////////////
// KMEANS class
////////////////////////////////
template <typename NumericalType>
class KMeans
{
    KMeans(const KMeans &other);
    KMeans &operator=(const KMeans &other);

public:
    // for convenience
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXt;
    typedef Eigen::Matrix<NumericalType, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXt;

    // ctor
    KMeans(const uint dim, const uint k, const uint initk,
           const NumericalType breakScore = (NumericalType)0.00001,
           const uint breakLoops = UINT_MAX) : mDim(dim),
                                               mK(k),
                                               mInitK(initk),
                                               mBreakScore(breakScore * breakScore),
                                               mBreakLoops(breakLoops)
    { }

    // compute centroids
    void compute(MatrixXt *centroids,
                 std::vector<uint> *freq,
                 const MatrixXt &indata)
    {
        centroids->setZero();
        freq->clear();

        // some checks
        if (indata.rows() != mDim)
        {
            std::cerr << "KMEANS: expecting input matrix to have the following dimensions: m(rows) = dim, n(cols) = number of input vectors\n";
            return;
        }
        if (centroids->rows() != mDim || centroids->cols() != mK)
        {
            std::cerr << "KMEANS: expecting centroid matrix to have the following dimensions: m(rows) = dim, n(cols) = k\n";
            return;
        }

        // compute initial seeds
        seeds(centroids, indata);

        // actual k means
        freq->clear();
        freq->resize(mK, 0);
        kmeans(centroids, freq, indata);
    }

private:
    void kmeans(MatrixXt *centroids,
                std::vector<uint> *freq,
                const MatrixXt &indata)
    {
        // frequencies
        MatrixXt tmpCenters(mDim, mK);
        VectorXt cmp(mDim);

        // kmeans algo
        const uint size = indata.cols();
        uint rounds = 0;
        for ( ; rounds < mBreakLoops; ++rounds)
        {
            // prepare round
            tmpCenters.setZero();
            std::fill(freq->begin(), freq->end(), 0);

            // iterate data points
            for (uint i = 0; i < size; ++i)
            {
                // grab vector and get index of closest seed
                const VectorXt &vec = indata.col(i);
                const uint seedID = idxOfClosestSeed(*centroids, vec);
                // remember point
                tmpCenters.col(seedID) += vec;
                freq->at(seedID) += 1u;
            }

            // recompute seeds
            bool change = false;
            for (uint s = 0; s < mK; ++s)
            {
                cmp = centroids->col(s);
                const uint f = freq->at(s);
                if (f != 0)
                {
                    centroids->col(s) = tmpCenters.col(s) / (NumericalType)f;
                }
                const NumericalType err = (centroids->col(s) - cmp).squaredNorm();
                if (err > mBreakScore)
                {
                    change = true;
                }
                //std::cerr << "centroid " << s << ": " << cmp.transpose() << ", supporters = " << f << ", err = " << std::sqrt(err) << std::endl;
            }
            std::cerr << "k-means round: " << rounds << std::endl;

            // nothing changed?
            if (!change) break;
        }
        std::cerr << "done." << std::endl;
    }

    // select initial seed configuration
    void seeds(MatrixXt *centroids,
               const MatrixXt &indata)
    {
        // draw initial k random samples
        MatrixXt initSamples(mDim, mInitK);
        const uint cols = indata.cols();
        for (uint i = 0; i < mInitK; ++i)
        {
            const uint idx = mRnd.rand() % cols;
            initSamples.col(i) = indata.col(idx);
        }

        // how many samples to throw away?
        const uint size = mInitK - mK;
        std::vector<bool> eliminated(mInitK, false);
        for (uint s = 0; s < size; ++s)
        {
            // eliminate seeds that are close to other seeds
            NumericalType dist = FLT_MAX;
            uint idx = 0;
            for (uint i = 0; i < mInitK; ++i)
            {
                // already eliminated
                if (eliminated[i]) continue;

                // find closest distance to all other seeds
                const NumericalType d = distToClosestSeed(initSamples, eliminated, i);
                if (d < dist)
                {
                    dist = d;
                    idx  = i;
                }
            }
            // mark initseed as used
            eliminated[idx] = true;
        }

        // store remaining seeds
        uint cnt = 0;
        for (uint i = 0; i < mK; ++i, ++cnt)
        {
            while (eliminated[cnt]) ++cnt;
            centroids->col(i) = initSamples.col(cnt);
        }
    }

    // dist to closest seed
    NumericalType distToClosestSeed(const MatrixXt &seeds, const std::vector<bool> &eliminated, const uint index) const
    {
        NumericalType d = FLT_MAX;
        const VectorXt &vec = seeds.col(index);
        const uint size = seeds.cols();
        for (uint i = 0; i < size; ++i)
        {
            if (i == index || eliminated[i]) continue;
            const NumericalType s = (vec - seeds.col(i)).squaredNorm();
            if (s < d)
            {
                d = s;
            }
        }
        return d;
    }

    // get index of closest seed
    uint idxOfClosestSeed(const MatrixXt &seeds, const VectorXt &vec) const
    {
        NumericalType d = FLT_MAX;
        uint idx = 0;
        const uint size = seeds.cols();
        for (uint i = 0; i < size; ++i)
        {
            const NumericalType s = (vec - seeds.col(i)).squaredNorm();
            if (s < d)
            {
                d = s;
                idx = i;
            }
        }
        return idx;
    }

    // dimensions and number of centroids
    uint mDim, mK, mInitK;
    NumericalType mBreakScore;
    uint mBreakLoops;
    XorShift<NumericalType> mRnd;
};


} // namespace

#endif
