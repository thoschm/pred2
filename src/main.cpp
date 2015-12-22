
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



#define DIM 3u
#define SAMPLES 6000u
#define K 100u


int main(int argc, char **argv)
{
    KMeans<float>::MatrixXt mat = KMeans<float>::MatrixXt::Random(DIM, SAMPLES);
    dumpMatrix(mat, "matrix.txt");

    WhiteningTransform<float> wt(DIM);
    PCAWhitening<float> pca(DIM, 0.0f, 0.0f);
    pca.computeTransform(&wt, mat);
    pca.applyTransformInPlace(&mat, wt);

    dumpMatrix(mat, "white.txt");

    KMeans<float> km(DIM, K, 10 * K);

    KMeans<float>::MatrixXt centroids(DIM, K);
    std::vector<uint> freq;
    km.compute(&centroids, &freq, mat);

    dumpMatrix(centroids, "centroids.txt");

    return 0;
}
