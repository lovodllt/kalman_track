#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

typedef Eigen::Matrix<float, 8, 1> Vector8f;
typedef Eigen::Matrix<float, 8, 8> Matrix8f;

/* x = [x,
 *      y,
 *      w,
 *      h,
 *      vx,
 *      vy,
 *      vw,
 *      vh]
 */

class KalmanFilter_ {
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
    int predictions_without_update = 0;

public:
    KalmanFilter_(float dt) : dt(dt)
    {
        //初始化状态向量
        x << 0, 0, 0, 0, 0, 0, 0, 0;

        //初始化状态转移矩阵
        F << 1, 0, 0, 0, dt, 0,  0,  0,
             0, 1, 0, 0, 0,  dt, 0,  0,
             0, 0, 1, 0, 0,  0,  dt, 0,
             0, 0, 0, 1, 0,  0,  0,  dt,
             0, 0, 0, 0, 1,  0,  0,  0,
             0, 0, 0, 0, 0,  1,  0,  0,
             0, 0, 0, 0, 0,  0,  1,  0,
             0, 0, 0, 0, 0,  0,  0,  1;

        //初始化过程噪声协方差矩阵
        Q = Matrix8f::Zero(8,8);
        Q.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5;

        //初始化观测噪声协方差矩阵
        R = Eigen::Matrix4f::Identity() * 0.1;

        //初始化误差协方差
        P = Matrix8f::Identity() * 10;

        //初始化观测矩阵
        H << 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0;
    }

    void init(const Eigen::Vector4f& z)
    {
        x.head<4>() = z;
        is_init = true;
    }

    // 预测
    void predict()
    {
        if(!is_init) return;
        x = F * x;
        P = F * P * F.transpose() + Q;
        predictions_without_update++;
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
        predictions_without_update = 0;
    }

    // 获取当前位置
    Eigen::Vector4f getPosition()
    {
        return x.head<4>();
    }

    int getPredictionsWithoutUpdate() const { return predictions_without_update; }
};