
#include <kmeans.h>
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




int main(int argc, char **argv)
{
    KMeans<float> km(2u, 3u, 100u);
    KMeans<float>::MatrixXt mat(2, 6000);

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


    KMeans<float>::MatrixXt centroids(2u, 3u);
    std::vector<uint> freq;

    km.compute(&centroids, &freq, mat);
    dumpMatrix(mat, "points.txt");
    dumpMatrix(centroids, "center.txt");
    std::cerr << centroids << std::endl;




    return 0;
}
