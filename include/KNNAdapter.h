#ifndef _KNNADAPTER_H_
#define _KNNADAPTER_H_


////////////////////////////////////////
// INCLUDES
////////////////////////////////////////
#include "_nanoflann.hpp"
#include <Eigen/Dense>


////////////////////////////////////////
// NAMESPACE
////////////////////////////////////////
namespace BOF
{
////////////////////////////////////////
// GenericKNNAdapter
////////////////////////////////////////
template <int VecDim,
          typename Container,
          typename ItemType,
          typename Accessor,
          typename NumericalType>
class GenericKNNAdapter;


template <int VecDim,
          typename Container,
          typename ItemType,
          typename Accessor,
          typename NumericalType>
struct KDTree
{
    typedef nanoflann::KDTreeSingleIndexAdaptor
            <nanoflann::L2_Adaptor<NumericalType, GenericKNNAdapter<VecDim, Container, ItemType, Accessor, NumericalType> >,
            GenericKNNAdapter<VecDim, Container, ItemType, Accessor, NumericalType>, VecDim, uint> type;
};


///////////////////////////////////////////////////
// VecDim...: number of dimensions of an item
//            -1 for dynamic allocation at run-time
//            (give attach() the correct dim)
// Container: e.g. std::vector<item>
//            must implement size(), operator[]
// Accessor.: must have a static function
//            at(item, dim) to get n-th element
// NumericalType: double / float / ...
///////////////////////////////////////////////////
template <int VecDim,
          typename Container,
          typename ItemType,
          typename Accessor,
          typename NumericalType>
class GenericKNNAdapter
{
private:
    GenericKNNAdapter(const GenericKNNAdapter &other);
    GenericKNNAdapter &operator=(const GenericKNNAdapter &other);

public:
    // containers for search results (index, distance)
    typedef std::pair<uint, NumericalType> IdxDistPair;
    typedef std::vector<IdxDistPair> IdxDistPairs;

    // ctor, dtor
    GenericKNNAdapter() : mList(NULL), mTree(NULL), mDim(-1) { }
    ~GenericKNNAdapter()
    {
        if (mTree != NULL) delete mTree;
    }

    // attach a container, note that the
    // actual data is not stored here
    // if VecDim is -1, you MUST specify "allocDim" here
    void attach(const Container *list, int allocDim = VecDim)
    {
        mDim = allocDim;
        mList = list;
        // rebuild tree
        rebuild();
    }

    // refresh tree, when data has changed
    void rebuild()
    {
        if (mTree != NULL) delete mTree;
        mTree = new typename KDTree<VecDim,
                                    Container,
                                    ItemType,
                                    Accessor,
                                    NumericalType>::type(mDim,
                                                         *this,
                                                         nanoflann::KDTreeSingleIndexAdaptorParams(10));
        mTree->buildIndex();
    }

    // has tree?
    bool hasTree() const
    {
        return mTree != NULL;
    }

    // get neighbors within squared radius
    const IdxDistPairs &radiusSearch(const ItemType &item,
                                     NumericalType radiusSqr)
    {
        mPairs.clear();
        nanoflann::SearchParams params;
        params.sorted = false;
        NumericalType *pt = new NumericalType[mDim];
        for (int i = 0; i < mDim; ++i)
        {
            pt[i] = Accessor::at(item, i);
        }
        mTree->radiusSearch(pt, radiusSqr, mPairs, params);
        delete[] pt;
        return mPairs;
    }

    // get pairs of indices and Eu. distances for K nearest neighbors
    IdxDistPairs nnSearchNSqrt(const ItemType &item,
                               uint K) const
    {
        IdxDistPairs res;
        res.reserve(K);
        uint *idx = new uint[K];
        NumericalType *dist = new NumericalType[K];
        NumericalType *pt = new NumericalType[mDim];
        for (int i = 0; i < mDim; ++i)
        {
            pt[i] = Accessor::at(item, i);
        }
        mTree->knnSearch(pt, K, idx, dist);
        for (uint i = 0; i < K; ++i)
        {
            res.push_back(std::make_pair(idx[i], std::sqrt(dist[i])));
        }
        delete[] pt;
        delete[] idx;
        delete[] dist;
        return res;
    }

    // ...stripped down for publication...

    // get direct access to tree
    const typename KDTree<VecDim, Container, ItemType, Accessor, NumericalType>::type *getTree()
    {
        return mTree;
    }

    // now some nanoflann stuff
    uint kdtree_get_point_count() const { return mList->cols(); } // TODO: this is crap: .cols()
    NumericalType kdtree_get_pt(uint idx, int dim) const
    {
        return Accessor::at((*mList).col(idx), dim); // TODO: this is crap: .col(idx)
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const { return false; }

private:
    // pointer to data
    const Container *mList;

    // actual k-d-tree
    typename KDTree<VecDim,
                    Container,
                    ItemType,
                    Accessor,
                    NumericalType>::type *mTree;
    int mDim;
    IdxDistPairs mPairs;
};


// namespace
}

#endif
