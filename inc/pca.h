#include<iostream>
#include<algorithm>
#include<cstdlib>
#include<fstream>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

class Pca {
private:

public:
    MatrixXd result;
    Pca(MatrixXd X);
    void featurenormalize(MatrixXd &X);
    void computeCov(MatrixXd &X, MatrixXd &C);
    void computeEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val);
    int computeDim(MatrixXd &val);

};