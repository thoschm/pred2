

#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <kmeans.h>
#include <whitening.h>
#include <histogram.h>
#include <fstream>
#include <collector.h>
#include <norm.h>
#include <wnnc.h>
#include <resize.h>

#include <interface.h>


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


#define SAMPLES 20000u

/*
#define K 10u
#define WINDOW 1031
#define AHEAD 100
#define FEATURE 32
#define WAVELET 8
#define PARTS 10

#define INDEX 12500
*/
uint labelfunc(const std::vector<float> &data, const uint last, const uint ahead)
{
    if (data[last] < data[ahead]) return 1u;
    return 0;
}


int main(int argc, char **argv)
{
    std::vector<float> indata, interp;
    //loadSequence(&indata, "chart.txt");
    for (uint i = 0; i < SAMPLES; ++i)
    {
        //indata.push_back((i % 7 == 0) ? 2.0 : 5.0);
        indata.push_back(sqrt(i) + std::sin(0.1 * i) + std::sin(0.05 * (i + 17)) * std::cos(0.02 * (i + 23)) + 0.01f * i + 5.0f * std::sin(0.01f * (i + 100)));
        //indata.push_back(i);
    }


    const uint halfsize = 0.7 * indata.size();
    std::vector<float> traindata(indata.begin(), indata.begin() + halfsize);
    std::vector<float> veridata(indata.begin() + halfsize, indata.end());
    dumpSequence(traindata, "traindata.txt");
    dumpSequence(veridata, "veridata.txt");

    BOFParameters bp;
    BOFClassifier<float>::MatrixXt words(bp.featureSize, bp.codeWords);
    BOFClassifier<float> clsf(bp);
    BOFClassifier<float>::MatrixXt sigs;
    NormParams<float> normparams(bp.featureSize);
    WhiteningTransform<float> whiteningtf(bp.featureSize);
    BOFClassifier<float>::MatrixXt features;

    clsf.codeWords(&words, &normparams, &whiteningtf, traindata);
    clsf.signatures(&sigs, words, normparams, whiteningtf, traindata);

    NormParams<float> np2(sigs.rows());
    WhiteningTransform<float> wtf2(sigs.rows());
    clsf.computeNormWhite(&sigs, &np2, &wtf2, sigs.rows());
   // std::cout << sigs.transpose() << std::endl;

    std::vector<uint> label;
    clsf.labels(&label, traindata, labelfunc);

    WeightedNNClassifier<float> wnnc(bp.codeWords * bp.numParts, 2u, 9u);
    wnnc.attach(&sigs, &label);
    std::cerr << "train..." << std::endl;
    wnnc.train();
    std::cerr << "dump..." << std::endl;
    wnnc.dump("space.txt");


    const uint limit = veridata.size() - bp.windowSize - bp.lookAhead;
    std::vector<float> outvec(veridata.size(), 0.0f),
                       wrong(veridata.size(), 0.0f);
    uint all = 0,
         correct = 0;
    for (uint i = 0; i <= limit; ++i)
    {
        BOFClassifier<float>::MatrixXt sig;
        clsf.signature(&sig, words, normparams, whiteningtf, veridata, i);
        clsf.forwardNormWhite(&sig, np2, wtf2, sigs.rows());

        BOFClassifier<float>::VectorXt tmp = sig.col(0);
        //std::cout << tmp.transpose() << std::endl;
        ClassificationResult<float> res = wnnc.classify(tmp);
        std::cout << res.category << " " << res.confidence;
        //if (res.confidence < 95.0) continue;
        if (veridata[i + bp.windowSize - 1] < veridata[i + bp.windowSize + bp.lookAhead - 1])
        {
            if (res.category == 1u)
            {
                std::cout << "!";
                outvec[i + bp.windowSize + bp.lookAhead - 1] = veridata[i + bp.windowSize + bp.lookAhead - 1];
                ++correct;
            }
            else
            {
                wrong[i + bp.windowSize + bp.lookAhead - 1] = veridata[i + bp.windowSize + bp.lookAhead - 1];
            }
        }
        else
        {
            if (res.category == 0u)
            {
                std::cout << "!";
                outvec[i + bp.windowSize + bp.lookAhead - 1] = veridata[i + bp.windowSize + bp.lookAhead - 1];
                ++correct;
            }
            else
            {
                wrong[i + bp.windowSize + bp.lookAhead - 1] = veridata[i + bp.windowSize + bp.lookAhead - 1];
            }
        }
        ++all;
        std::cerr << std::endl;
    }
    std::cout << "CORRECT: " << correct << "/" << all << " (" << (100.0 * correct / all) << "%)" << std::endl;
    dumpSequence(outvec, "correct.txt");
    dumpSequence(wrong, "wrong.txt");


    PCAWhitening<float> pca(bp.featureSize);
    pca.inverseTransformInPlace(&words, whiteningtf);
    Normalization<float> elemnorm(bp.featureSize);
    elemnorm.inverseParamsInPlace(&words, normparams);


    float s[(int)bp.waveletType], w[(int)bp.waveletType];
    WaveletCoefficients<float>::lookup(bp.waveletType, s, w);
    FastDWT<float> dwt(bp.featureSize);
    std::ofstream ofs;
    ofs.open("words.txt", std::ios::out);
    for (uint k = 0; k < words.cols(); ++k)
    {
        dwt.inverse(words.col(k).data(), s, w, bp.waveletType);
        for (uint l = 0; l < bp.featureSize; ++l)
        {
            ofs << l << " " << words(l, k) << std::endl;
        }
        ofs << std::endl;
    }
    ofs.close();

    return 0;
}
