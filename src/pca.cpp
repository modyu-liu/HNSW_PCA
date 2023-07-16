#include "pca.h"
void Pca::featurenormalize(MatrixXd &X)
{
    //计算每一维度均值
    MatrixXd meanval = X.colwise().mean();
    RowVectorXd meanvecRow = meanval;
    //样本均值化为0
    X.rowwise() -= meanvecRow;
}
void Pca::computeCov(MatrixXd &X, MatrixXd &C)
{
    //计算协方差矩阵C = XTX / n-1;
    C = X.adjoint() * X;
    C = C.array() / (X.rows() - 1);
}
void Pca::computeEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)
{
    //计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列
    SelfAdjointEigenSolver<MatrixXd> eig(C);

    vec = eig.eigenvectors();
    val = eig.eigenvalues();
}
int Pca::computeDim(MatrixXd &val)
{
    int dim;
    double sum = 0;
    for (int i = val.rows() - 1; i >= 0; --i)
    {
        sum += val(i, 0);
        dim = i;

        if (sum / val.sum() >= 0.95)
            break;
    }
    return val.rows() - dim;
}

Pca::Pca(MatrixXd X){

    int n = X.rows();
    MatrixXd C(n, n);
    MatrixXd vec, val;
    featurenormalize(X);
    //计算协方差
    computeCov(X, C);
    //计算特征值和特征向量
    computeEig(C, vec, val);

    //计算损失率，确定降低维数
    int dim = computeDim(val);
    //计算结果
    // cout<<"find::"<<n<<' ' << X.cols()<<' '<<dim<<'\n';
    // cout<<vec.rows() << ' ' << vec.cols() <<'\n';

    MatrixXd res = X * vec.rightCols(dim);
    // cout<<"ok!"<<'\n';

    //输出结果
    this->result = res;

    //cout << "the result is " << res.rows() << "x" << res.cols() << " after pca algorithm." << endl;

}