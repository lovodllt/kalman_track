#include "kalman.h"

const string onnx_path = "../model/best.onnx";
const string video_path = "../data/test2.mp4";
const float fps = 40.0f;
const int t = static_cast<int>(1000/fps);
const float IOU_THRESHOLD = 0.5;

struct Bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float class_probability;
    int id;
};

struct Track
{
    int id = -1;                  //跟踪id
    KalmanFilter_ kf;             //卡尔曼滤波器
    int age = 0;                  //存活帧数
    int lost_frames = 0;          //丢失帧数
    bool confirmed = false;       //是否被确认
    bool is_deleted = false;      //是否被删除
};

vector<Track> tracks;   // 初始化跟踪器列表
int next_id = 1;          // 下一个分配的ID

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

vector<Bbox> nms(const vector<Bbox> &bboxes, float iouThreshold = 0.3)
{
    vector<Bbox> result;
    if (bboxes.empty()) return result;

    // 按照得分从高到低排序
    vector<pair<int, float>> scores;
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        scores.emplace_back(i, bboxes[i].score);
    }
    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    vector<bool> keep(bboxes.size(), true);

    for (int i = 0; i < scores.size(); i++)
    {
        int index = scores[i].first;
        if (!keep[index]) continue;

        result.push_back(bboxes[index]);

        for (int j = i + 1; j < scores.size(); j++)
        {
            int nextIndex = scores[j].first;
            if (keep[nextIndex])
            {
                float iouValue = iou(bboxes[index], bboxes[nextIndex]);
                if (iouValue > iouThreshold)
                {
                    keep[nextIndex] = false;
                }
            }
        }
    }
    return result;
}

vector<Bbox> postprocess(const Mat &output,float confThreshold=0.5)
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

    result = nms(bboxes);

    return result;
}

void drawBoxes(Mat &img, const vector<Bbox> &track_boxes)
{
    for(const auto &track_box : track_boxes)
    {
        rectangle(img,Point(track_box.x1,track_box.y1),Point(track_box.x2,track_box.y2),Scalar(0,255,0),2);
        //putText(img,"Person",Point(target.x1,target.y1-5),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0),2);
        putText(img,"ID:"+to_string(track_box.id),Point(track_box.x1,track_box.y1-10),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0),2);
    }
}

Mat resizeWithPadding(const Mat& img,Size targetSize)
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

/* 逻辑：
 * 1.若tracks为空，所有检测结果存入unmatched_detections，detections同理
 * 2.计算代价矩阵，当track与detection的IoU大于阈值时，视为匹配
 * 3.使用匈牙利算法匹配，assignment[i] = j 表示跟踪器 i 匹配到检测 j
 * 4.遍历匹配结果，将有效结果存入matches，用matched_track和matched_detection标记已匹配的track和detection
 * 5.处理未匹配的tracks和detections，存入unmatched_tracks和unmatched_detections
 * 6.更新匹配成功的tracks，调用update函数更新状态，同时重置lost_frames，增加age（大于5时标记为确认状态）
 * 7.对于未匹配的tracks，增加lost_frames，根据其确认状态，若lost_frames超过对应状态阈值，则标记为删除状态
 * 8.为未匹配的detections创建新的tracks，初始化状态，分配id，增加age
 * 9.清除已标记为删除的tracks
 */
void hungarianLoader(const vector<Bbox> &detections, vector<Track>& tracks, vector<pair<int,int>>& matches,
                    vector<int>& unmatched_tracks, vector<int>& unmatched_detections, float dt)
{
    // 清空匹配结果
    matches.clear();
    unmatched_tracks.clear();
    unmatched_detections.clear();

    if (detections.empty())
    {
        for(int i = 0; i < tracks.size(); i++)
        {
            unmatched_tracks.push_back(i);
        }
        for(int i : unmatched_tracks)
        {
            tracks[i].lost_frames++;
        }
        return;
    }

    // 级联匹配
    sort(tracks.begin(), tracks.end(), [](const Track& a, const Track& b) {
        return a.age > b.age;});

    for(auto& track : tracks)
    {
        track.kf.setF(dt);
        track.kf.predict();
    }

    // 计算cost矩阵(当track与detection的IoU大于阈值时，视为匹配)
    MatrixXb cost_matrix(tracks.size(), detections.size());
    for(int i = 0; i < tracks.size(); i++)
    {
        Eigen::Vector4f pos = tracks[i].kf.getPosition();
        Bbox track_box = {pos[0] - pos[2]/2.0f, pos[1] - pos[3]/2.0f, pos[0] + pos[2]/2.0f, pos[1] + pos[3]/2.0f};
        for(int j = 0; j < detections.size(); j++)
        {
            cost_matrix(i, j) = iou(track_box, detections[j]) > IOU_THRESHOLD;
        }
    }

    // 匈牙利算法匹配
    HungarianAlgorithm solver;
    Eigen::VectorXi assignment = solver.solve(cost_matrix);

    // 处理匹配结果
    vector<bool> matched_track(tracks.size(), false);
    vector<bool> matched_detection(detections.size(), false);

    for(int j = 0; j < assignment.size(); j++)
    {
        if(assignment[j] != -1)
        {
            int i = assignment[j];
            if(i >= 0 && i < tracks.size())
            {
                matches.emplace_back(i,j);
                matched_track[i] = true;
                matched_detection[j] = true;
            }
        }
    }

    // 处理未匹配的tracks和detections
    for(int i = 0; i < tracks.size(); i++)
    {
        if(!matched_track[i])
        {
            unmatched_tracks.push_back(i);
        }
    }
    for(int j = 0; j < detections.size(); j++)
    {
        if(!matched_detection[j])
        {
            unmatched_detections.push_back(j);
        }
    }

    // 更新匹配成功的tracks
    for(auto& match : matches)
    {
        int track_idx = match.first;
        int detection_idx = match.second;

        // 获取检测框中心和宽高
        auto& det = detections[detection_idx];
        float cx = (det.x1 + det.x2) / 2.0f;
        float cy = (det.y1 + det.y2) / 2.0f;
        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;

        tracks[track_idx].kf.update(Eigen::Vector4f(cx, cy, w, h));
        tracks[track_idx].lost_frames = 0; //重置跟踪器丢失帧数
        tracks[track_idx].age++;
        if(!tracks[track_idx].confirmed && tracks[track_idx].age > 3)
        {
            tracks[track_idx].confirmed = true;
            tracks[track_idx].id = next_id++;
        }

    }

    // 处理未匹配的tracks
    for(int i : unmatched_tracks)
    {
        tracks[i].lost_frames++;
    }

    // 创建新跟踪器
    for(int j : unmatched_detections)
    {
        auto& det = detections[j];
        float cx = (det.x1 + det.x2) / 2.0f;
        float cy = (det.y1 + det.y2) / 2.0f;
        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;

        Track new_track;
        new_track.kf.init(Eigen::Vector4f(cx, cy, w, h));
        new_track.age = 1;
        new_track.lost_frames = 0;
        tracks.push_back(new_track);
    }

    // 删除丢失的tracks
    for(auto& track : tracks)
    {
        if(track.confirmed)
        {
            if(track.lost_frames > 20)
            {
                track.is_deleted = true;
            }
        }
        else
        {
            if(track.lost_frames > 0)
            {
                track.is_deleted = true;
            }
        }
    }
    tracks.erase(remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.is_deleted; }), tracks.end());
}

int main()
{
    // 加载模型
    dnn::Net net = dnn::readNetFromONNX(onnx_path);

    VideoCapture cap(video_path);

    auto last_time = chrono::steady_clock::now();

    while(true)
    {
        auto current_time = chrono::steady_clock::now();
        float dt = chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        if(dt == 0) continue;

        Mat img;
        cap >> img;

        img = resizeWithPadding(img, Size(640, 640));

        Mat blob = dnn::blobFromImage(img,1.0/255.0,Size(640,640),Scalar(0,0,0),true);
        net.setInput(blob);

        vector<Mat> outputs;
        vector<String> outNames={"output0"};
        net.forward(outputs,outNames);

        vector<Bbox> detections;
        for(auto &output:outputs)
        {
            vector<Bbox> currentBoxes = postprocess(output);
            detections.insert(detections.end(), currentBoxes.begin(), currentBoxes.end());
        }

        vector<pair<int,int>> matches;
        vector<int> unmatched_tracks, unmatched_detections;
        hungarianLoader(detections,tracks,matches,unmatched_tracks,unmatched_detections, dt);

        for (auto& track : tracks)
        {
            Eigen::Vector4f pos = track.kf.getPosition();
            float x1 = pos[0]-pos[2]/2;
            float y1 = pos[1]-pos[3]/2;
            float x2 = pos[0]+pos[2]/2;
            float y2 = pos[1]+pos[3]/2;

            if (track.confirmed)
            {
                rectangle(img, Point2f(x1,y1), Point2f(x2,y2), Scalar(0,255,0), 2);
                putText(img, "ID:"+to_string(track.id), Point2f(x1,y1-10),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
            }
        }

        imshow("img", img);
        waitKey(t);
    }
}
