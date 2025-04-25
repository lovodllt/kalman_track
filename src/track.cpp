#include "kalman.h"

using namespace std;
using namespace cv;

const string onnx_path = "/home/lovod/CLionProjects/src/track/model/best.onnx";
const string video_path = "/home/lovod/CLionProjects/src/track/data/test2.mp4";
const float fps = 40;
const int t = static_cast<int>(1000/fps);
const float dt = 1 / fps;

// 初始化卡尔曼滤波器
KalmanFilter_ kf(dt);

struct Bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float class_probability;
};

void coordinate_convert(float *pdata, vector<Bbox> &bboxes)
{
    Bbox bbox;
    float x,y,w,h;
    x = pdata[0];
    y = pdata[1];
    w = pdata[2];
    h = pdata[3];
    bbox.x1 = x - w / 2;
    bbox.y1 = y - h / 2;
    bbox.x2 = x + w / 2;
    bbox.y2 = y + h / 2;
    bbox.score = pdata[4];
    bbox.class_probability = pdata[5];
    bboxes.push_back(bbox);
}

float iou(const Bbox &box1, const Bbox &box2)
{
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    float x1 = max(box1.x1, box2.x1);
    float y1 = max(box1.y1, box2.y1);
    float x2 = min(box1.x2, box2.x2);
    float y2 = min(box1.y2, box2.y2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = area1 + area2 - intersection;
    return intersection / union_area;
}

vector<Bbox> nms(const vector<Bbox> &bboxes, float iouThreshold = 0.4)
{
    vector<Bbox> result;
    if (bboxes.empty()) return result;

    // 按照得分从高到低排序
    vector<pair<int, float>> scores;
    for (size_t i = 0; i < bboxes.size(); ++i) {
        scores.emplace_back(i, bboxes[i].score);
    }
    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    vector<bool> keep(bboxes.size(), true);

    for (size_t i = 0; i < scores.size(); ++i) {
        int index = scores[i].first;
        if (!keep[index]) continue;

        result.push_back(bboxes[index]);

        for (size_t j = i + 1; j < scores.size(); ++j) {
            int nextIndex = scores[j].first;
            if (keep[nextIndex]) {
                float iouValue = iou(bboxes[index], bboxes[nextIndex]);
                if (iouValue > iouThreshold) {
                    keep[nextIndex] = false;
                }
            }
        }
    }
    return result;
}

vector<Bbox> postprocess(const Mat &output,float confThreshold=0.4)
{
    vector<Bbox> bboxes;
    vector<Bbox> result;
    float *pdata = (float*)output.data; //pdata指向output的第一个元素,可像数组一样访问
    int length = 6;

    for(int i=0;i<output.total()/length;i++)
    {
        double confidence = pdata[4];
        if(confidence > confThreshold)
        {
            coordinate_convert(pdata,bboxes);
        }
        pdata += length;
    }

    result = nms(bboxes,0.25);

    return result;
}

void drawBoxes(Mat &img, const vector<Bbox> &bboxes)
{
    for(const auto &box:bboxes)
    {
        rectangle(img,Point(box.x1,box.y1),Point(box.x2,box.y2),Scalar(0,255,0),2);
        putText(img,"Person",Point(box.x1,box.y1-5),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0),2);
    }
}

Mat resizeWithPadding(const cv::Mat& img, cv::Size targetSize)
{
    int h = img.rows;
    int w = img.cols;
    int th = targetSize.height;
    int tw = targetSize.width;

    // 计算缩放比例
    float scale = min(static_cast<float>(tw) / w, static_cast<float>(th) / h);
    int newW = static_cast<int>(w * scale);
    int newH = static_cast<int>(h * scale);

    // 缩放图像
    Mat resizedImage;
    resize(img, resizedImage, cv::Size(newW, newH));

    // 计算填充量
    int padW = tw - newW;
    int padH = th - newH;

    // 计算左右和上下的填充量
    int left = padW / 2;
    int right = padW - left;
    int top = padH / 2;
    int bottom = padH - top;

    // 填充图像
    Mat paddedImage;
    copyMakeBorder(resizedImage, paddedImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return paddedImage;
}

void kalmanLoader(Mat &img,vector<Bbox> &boxes)
{
    if (boxes.empty())
    {
        kf.predict();
        if (kf.getPredictionsWithoutUpdate() > 5)
        {
            kf = KalmanFilter_(dt);  // 重置滤波器
        }
    }
    else
    {
        for (auto &box : boxes)
        {
            // 计算中心点、宽高
            float cx = (box.x1 + box.x2) / 2;
            float cy = (box.y1 + box.y2) / 2;
            float w = box.x2 - box.x1;
            float h = box.y2 - box.y1;

            // 调用卡尔曼滤波
            kf.predict();
            kf.update(Eigen::Vector4f(cx, cy, w, h));

            // 更新框坐标
            Eigen::Vector4f state = kf.getPosition();
            box.x1 = state[0] - state[2]/2.0f;
            box.y1 = state[1] - state[3]/2.0f;
            box.x2 = state[0] + state[2]/2.0f;
            box.y2 = state[1] + state[3]/2.0f;
        }
    }
}

int main()
{
    // 加载模型
    dnn::Net net = dnn::readNetFromONNX(onnx_path);

    VideoCapture cap(video_path);

    while(true)
    {
        Mat img;
        cap >> img;

        //resize(img, img, Size(640, 640));
        img = resizeWithPadding(img, Size(640, 640));

        Mat blob = dnn::blobFromImage(img,1.0/255.0,Size(640,640),Scalar(0,0,0),true);
        net.setInput(blob);

        vector<Mat> outputs;
        vector<String> outNames={"output0"};
        net.forward(outputs,outNames);

        vector<Bbox> boxes;
        for(auto &output:outputs)
        {
            vector<Bbox> currentBoxes = postprocess(output, 0.4);
            boxes.insert(boxes.end(), currentBoxes.begin(), currentBoxes.end());
        }

        kalmanLoader(img,boxes);

        drawBoxes(img, boxes);

        imshow("img", img);
        waitKey(t);
    }
}
