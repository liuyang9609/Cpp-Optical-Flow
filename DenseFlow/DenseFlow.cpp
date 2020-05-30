#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap("highway.mov");

    Mat flow, frame;
    UMat flowUmat, prevgray;

    for (;;)
    {
        bool Is = cap.grab();
        if (Is == false) {
            cout << "Video Capture Fail" << endl;
            break;
        }
        else {
            Mat img;
            Mat original;

            cap.retrieve(img, 0);
            resize(img, img, Size(640, 480));

            img.copyTo(original);
            cvtColor(img, img, COLOR_BGR2GRAY);

            if (prevgray.empty() == false) {
                calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 48, 2, 8, 1.2, 0);
                flowUmat.copyTo(flow);

                for (int y = 0; y < original.rows; y += 5) {
                    for (int x = 0; x < original.cols; x += 5) {
                        const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
                        line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
                        circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);
                    }
                }

                namedWindow("prew", WINDOW_AUTOSIZE);
                imshow("prew", original);

                img.copyTo(prevgray);
            }
            else {
                img.copyTo(prevgray);
            }

            int key1 = waitKey(20);
        }
    }
}
