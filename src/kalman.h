#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef Eigen::Matrix<float, 8, 1> Vector8f;
typedef Eigen::Matrix<float, 8, 8> Matrix8f;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

class KalmanFilter_ {
/* x = [x,
 *      y,
 *      w,
 *      h,
 *      vx,
 *      vy,
 *      vw,
 *      vh]
 */
private:
    Vector8f x;//状态向量
    Matrix8f F;//状态转移矩阵
    Matrix8f P;//误差协方差矩阵
    Matrix8f Q;//过程噪声协方差矩阵
    Eigen::Matrix4f R;//观测噪声协方差矩阵
    Eigen::Matrix<float,8,4> K;//卡尔曼增益
    Eigen::Matrix<float,4,8> H;//观测矩阵
    float dt;
    bool is_init = false;

public:
    KalmanFilter_() : dt(0.01)
    {
        //初始化状态向量
        x << 0, 0, 0, 0, 0, 0, 0, 0;

        //初始化状态转移矩阵
        F = Matrix8f::Identity();

        //初始化过程噪声协方差矩阵
        Q = Matrix8f::Zero(8,8);
        Q.diagonal() << 1e-2, 1e-2, 1e-2, 2e-2, 5e-2, 5e-2, 1e-4, 4e-2;

        //初始化观测噪声协方差矩阵
        R = Eigen::Matrix4f::Identity() * (1e-2);

        //初始化误差协方差
        P = Matrix8f::Identity() * 10;

        //初始化观测矩阵
        H << 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0;
    }

    void setF(float dt)
    {
        F << 1, 0, 0, 0, dt, 0,  0,  0,
             0, 1, 0, 0, 0,  dt, 0,  0,
             0, 0, 1, 0, 0,  0,  dt, 0,
             0, 0, 0, 1, 0,  0,  0,  dt,
             0, 0, 0, 0, 1,  0,  0,  0,
             0, 0, 0, 0, 0,  1,  0,  0,
             0, 0, 0, 0, 0,  0,  1,  0,
             0, 0, 0, 0, 0,  0,  0,  1;

        Vector8f new_x = x;
        x[0] = new_x[0] + new_x[4] * dt;
        x[1] = new_x[1] + new_x[5] * dt;
        x[2] = new_x[2] + new_x[6] * dt;
        x[3] = new_x[3] + new_x[7] * dt;
    }

    void init(const Eigen::Vector4f& z)
    {
        x.head<4>() = z;
        is_init = true;
    }

    // 预测
    void predict(float dt)
    {
        if(!is_init) return;

        setF(dt);
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    // 更新
    void update(const Eigen::Vector4f& z)
    {
        if(!is_init) init(z);

        auto y = z - H * x;
        auto S = H * P * H.transpose() + R;
        K = P * H.transpose() * S.inverse();

        x = x + K * y;
        P = (Matrix8f::Identity() - K * H) * P;
    }

    // 获取当前位置
    Eigen::Vector4f getPosition()
    {
        return x.head<4>();
    }
};

class HungarianAlgorithm{
public:
    Eigen::VectorXi solve(const MatrixXb& cost_matrix)
    {
        int n = cost_matrix.rows(); //代价矩阵的行数（跟踪器数量）
        int m = cost_matrix.cols(); //代价矩阵的列数（检测框数量）
        Eigen::VectorXi assignment = Eigen::VectorXi::Constant(n,-1); //存储每个跟踪器的匹配结果
        Eigen::VectorXi visited = Eigen::VectorXi::Zero(m); //列访问标记

        //匹配跟踪器和检测框
        for(int i = 0;i < n;i++)
        {
            visited.setZero();
            dfs(i, cost_matrix, assignment, visited);
        }

        return assignment;
    }

private:
    /*逻辑：
     * 1. 遍历每个跟踪器，对每个跟踪器，遍历每个检测框
     * 2. 如果cost_matrix[i,j]为true，且检测框j未被访问过，则直接占用该匹配
     * 3. 如果检测框j已被匹配，则递归调用dfs函数，尝试寻找其他匹配
     * 4. 如果找到匹配，则返回true，否则返回false
     */
    //深度优先搜索匹配跟踪器和检测框
    bool dfs(int i, const MatrixXb& cost_matrix, Eigen::VectorXi& assignment, Eigen::VectorXi& visited)
    {
        for(int j = 0;j < cost_matrix.cols();j++)
        {
            if(cost_matrix(i,j) && !visited(j))
            {
                visited(j) = 1;
                if(assignment(j) == -1 || dfs(assignment(j), cost_matrix, assignment, visited))
                {
                    assignment(j) = i;
                    return true;
                }
            }
        }
        return false;
    }
};