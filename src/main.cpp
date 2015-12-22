
#include <kmeans.h>
#include <whitening.h>
#include <fstream>


using namespace BOF;


bool loadSequence(std::vector<float> *seq, const char *file)
{
    seq->clear();
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail())
    {
        std::cerr << "failed to read sequence!\n";
        return false;
    }
    float val;
    for ( ; ; )
    {
        ifs >> val;
        if (ifs.eof()) break;
        seq->push_back(val);
    }
    ifs.close();
    return true;
}


bool dumpSequence(const std::vector<float> &seq, const char *file)
{
    std::ofstream ofs;
    ofs.open(file, std::ios::out);
    if (ofs.fail())
    {
        std::cerr << "failed to write sequence!\n";
        return false;
    }
    for (uint i = 0; i < seq.size(); ++i)
    {
        ofs << seq[i] << std::endl;
    }
    ofs.close();
    return true;
}


bool dumpMatrix(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &mat, const char *file)
{
    std::ofstream ofs;
    ofs.open(file, std::ios::out);
    if (ofs.fail())
    {
        std::cerr << "failed to write sequence!\n";
        return false;
    }
    for (uint i = 0; i < mat.cols(); ++i)
    {
        for (uint k = 0; k < mat.rows(); ++k)
        {
            ofs << mat(k, i) << " ";
        }
        ofs << std::endl;
    }
    ofs.close();
    return true;
}



#define DIM 2u
#define SAMPLES 6000u


int main(int argc, char **argv)
{
    KMeans<float>::MatrixXt mat(DIM, SAMPLES);

    NormalDistGenerator<float> nd;
    for (uint i = 0; i < 2000; ++i)
    {
        for (uint k = 0; k < 2u; ++k)
        {
            mat.col(i)(k) = nd.rand();
        }
    }
    for (uint i = 2000; i < 4000; ++i)
    {
        for (uint k = 0; k < 2u; ++k)
        {
            mat.col(i)(k) = nd.rand() + 5.0f;
        }
    }
    for (uint i = 4000; i < 6000; ++i)
    {
        for (uint k = 0; k < 2u; ++k)
        {
            mat.col(i)(k) = nd.rand() + 10.0f;
        }
    }
    dumpMatrix(mat, "matrix.txt");



    WhiteningTransform<float> wt(DIM);
    PCAWhitening<float> pca(DIM);
    pca.computeTransform(&wt, mat);
    pca.applyTransformInPlace(&mat, wt);

    dumpMatrix(mat, "white.txt");



    return 0;
}
