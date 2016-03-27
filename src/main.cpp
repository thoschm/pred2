

#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <kmeans.h>
#include <whitening.h>
#include <histogram.h>
#include <fstream>
#include <collector.h>
#include <norm.h>
#include <wnnc.h>
#include <resize.h>


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


#define SAMPLES 2000u
#define K 10u
#define WINDOW 1031
#define AHEAD 100
#define FEATURE 32
#define WAVELET 8
#define PARTS 10

#define INDEX 12500


int main(int argc, char **argv)
{
    std::vector<float> indata, interp;
    loadSequence(&indata, "chart.txt");
    for (uint i = 0; i < SAMPLES; ++i)
    {
        //indata.push_back((i % 7 == 0) ? 2.0 : 5.0);
        //indata.push_back(sqrt(i) + std::sin(0.1 * i) + std::sin(0.05 * (i + 17)) * std::cos(0.02 * (i + 23)) + 0.01f * i + 5.0f * std::sin(0.01f * (i + 100)));
        //indata.push_back(i);
    }

    dumpSequence(indata, "seq.txt");
    Interpolator<float>::resize(&interp, 5000u, indata, LANCZOS4);
    dumpSequence(interp, "interp.txt");


    return 0;



    const uint halfsize = 0.7 * indata.size();
    std::vector<float> traindata = indata;//(indata.begin(), indata.begin() + halfsize);
    std::vector<float> veridata = indata;//(indata.begin() + halfsize, indata.end());
    dumpSequence(traindata, "traindata.txt");
    dumpSequence(veridata, "veridata.txt");


    SeriesCollector<float> collector(WINDOW, FEATURE, K, PARTS, (WaveletType)WAVELET);
    SeriesCollector<float>::MatrixXt features;
    SeriesCollector<float>::MatrixXt words(FEATURE, K);

    WhiteningTransform<float> wt(FEATURE);
    NormParams<float> np(FEATURE);
    collector.append(&features, traindata);
    collector.codeWords(&words, &np, &wt, &features);
    std::cerr << words.transpose() << std::endl << std::endl;

    SeriesCollector<float>::VectorXt hist(K * PARTS);
    collector.signature(&hist, np, wt, indata, words, INDEX);
    std::cerr << "signature:\n" << hist.transpose() << std::endl;

    collector.dumpBasisActivation("sig.txt", "act.txt", indata, words, hist, np, wt, INDEX);


    // create labeled samples
    uint limit = traindata.size() - WINDOW - AHEAD;
    std::vector<Sample<float> > vec;
    for (uint i = 0; i <= limit; ++i)
    {
        vec.push_back(Sample<float>(K * PARTS));
        collector.signature(&(vec.back().signature), np, wt, traindata, words, i);
        std::cerr << ".";
        if (traindata[i + WINDOW - 1] < traindata[i + WINDOW + AHEAD - 1])
        {
            vec.back().label = 1;
        }
        else
        {
            vec.back().label = 0;
        }
    }
    std::cerr << std::endl;

    WeightedNNClassifier<float> wnnc(K * PARTS, 2u, 99u);
    wnnc.attach(&vec);
    wnnc.train();
    wnnc.dump("space.txt");

    limit = veridata.size() - WINDOW - AHEAD;
    std::vector<float> outvec(veridata.size(), 0.0f),
                       wrong(veridata.size(), 0.0f);
    uint all = 0,
         correct = 0;
    for (uint i = 0; i <= limit; ++i)
    {
        Sample<float> sa(K * PARTS);
        collector.signature(&(sa.signature), np, wt, veridata, words, i);
        ClassificationResult<float> res = wnnc.classify(sa);
        std::cout << res.category << " " << res.confidence;
        //if (res.confidence < 95.0) continue;
        if (veridata[i + WINDOW - 1] < veridata[i + WINDOW + AHEAD - 1])
        {
            if (res.category == 1u)
            {
                std::cerr << "!";
                outvec[i + WINDOW + AHEAD - 1] = veridata[i + WINDOW + AHEAD - 1];
                ++correct;
            }
            else
            {
                wrong[i + WINDOW + AHEAD - 1] = veridata[i + WINDOW + AHEAD - 1];
            }
        }
        else
        {
            if (res.category == 0u)
            {
                std::cerr << "!";
                outvec[i + WINDOW + AHEAD - 1] = veridata[i + WINDOW + AHEAD - 1];
                ++correct;
            }
            else
            {
                wrong[i + WINDOW + AHEAD - 1] = veridata[i + WINDOW + AHEAD - 1];
            }
        }
        ++all;
        std::cerr << std::endl;
    }
    std::cout << "CORRECT: " << correct << "/" << all << " (" << (100.0 * correct / all) << "%)" << std::endl;
    dumpSequence(outvec, "correct.txt");
    dumpSequence(wrong, "wrong.txt");

    std::ofstream ofs;
 /*   ofs.open("centroids.txt", std::ios::out);
    ofs << words.transpose() << std::endl;
    ofs.close();
*/


    PCAWhitening<float> pca(FEATURE);
    pca.inverseTransformInPlace(&words, wt);
    //std::cerr << words.transpose() << std::endl;
    Normalization<float> elemnorm(FEATURE);
    elemnorm.inverseParamsInPlace(&words, np);


    float s[WAVELET], w[WAVELET];
    WaveletCoefficients<float>::lookup((WaveletType)WAVELET, s, w);
    FastDWT<float> dwt(FEATURE);
    ofs.open("words.txt", std::ios::out);
    for (uint k = 0; k < K; ++k)
    {
        dwt.inverse(words.col(k).data(), s, w, WAVELET);
        for (uint l = 0; l < FEATURE; ++l)
        {
            ofs << l << " " << words(l, k) << std::endl;
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
