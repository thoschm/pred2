

#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <kmeans.h>
#include <whitening.h>
#include <histogram.h>
#include <fstream>
#include <collector.h>


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



#define SAMPLES 10u
#define K 50u
#define WINDOW 1000u
#define FEATURE 10u


int main(int argc, char **argv)
{

    std::vector<float> indata;
    loadSequence(&indata, "chart.txt");
    /*for (uint i = 0; i < SAMPLES; ++i)
    {
        //indata.push_back(std::sin(0.1 * i) + std::sin(0.05 * (i + 17)) * std::cos(0.02 * (i + 23)) + 0.01f * i + 5.0f * std::sin(0.01f * (i + 100)));
        indata.push_back(i);
    }*/

    SeriesCollector<float> collector(WINDOW, FEATURE, K);
    SeriesCollector<float>::MatrixXt words(FEATURE, K);

    WhiteningTransform<float> wt(FEATURE);
    collector.codeWords(&words, &wt, indata);
    std::cerr << words.transpose() << std::endl << std::endl;

    /*
    SeriesCollector<float>::VectorXt hist(K);
    collector.signature(&hist, wt, indata, words, 0);
    std::cerr << "signature:\n" << hist.transpose() << std::endl;
    collector.signature(&hist, wt, indata, words, 10);
    std::cerr << "signature:\n" << hist.transpose() << std::endl;
*/


    std::ofstream ofs;
 /*   ofs.open("centroids.txt", std::ios::out);
    ofs << words.transpose() << std::endl;
    ofs.close();
*/


    PCAWhitening<float> pca(FEATURE);
    pca.inverseTransformInPlace(&words, wt);
    std::cerr << words.transpose() << std::endl;

    ofs.open("words.txt", std::ios::out);
    for (uint k = 0; k < K; ++k)
    {
        for (uint l = 0; l < FEATURE - 1u; ++l)
        {
            ofs << (words(FEATURE - 1u, k) * (WINDOW - (FEATURE - 1u))) + l << " " << words(l, k) << std::endl;
        }
        ofs << std::endl;
    }
    ofs.close();


/*
    std::ofstream ofs;
    ofs.open("words.txt", std::ios::out);
    for (uint k = 0; k < K; ++k)
    {
        for (uint l = 0; l < FEATURE; ++l)
        {
            ofs << l << " " << words(l, k) << std::endl;
        }
        ofs << std::endl;
    }
    ofs.close();*/

/*
    NormalDistGenerator<float> n;
    KMeans<float>::MatrixXt mat(3, 2000);
    for (uint i = 0; i < 1000u; ++i)
    {
        for (uint k = 0; k < mat.rows(); ++k)
        {
            mat(k, i) = n.rand();
        }
    }
    for (uint i = 1000; i < 2000u; ++i)
    {
        for (uint k = 0; k < mat.rows(); ++k)
        {
            mat(k, i) = n.rand() + 10.0f;
        }
    }


    dumpMatrix(mat, "mat.txt");
    PCAWhitening<float> pca(3);
    WhiteningTransform<float> wt(3);
    pca.computeTransform(&wt, mat);
    pca.applyTransformInPlace(&mat, wt);

    KMeans<float> kmeans(3, 4, 50);
    std::vector<uint> freq;
    KMeans<float>::MatrixXt centers(3, 4);
    kmeans.compute(&centers, &freq, mat);
    dumpMatrix(centers, "wcenter.txt");

    dumpMatrix(mat, "white.txt");
    pca.inverseTransformInPlace(&centers, wt);
    dumpMatrix(centers, "invcenter.txt");

    */
    return 0;
}
