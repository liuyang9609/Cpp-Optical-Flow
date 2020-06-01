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
	Mat last_frame, next_frame;
	Mat flow_frame;

	last_frame = imread("000045_10.png");
	next_frame = imread("000045_11.png");

	Mat last_frame_gray, next_frame_gray;
	cvtColor(last_frame, last_frame_gray, COLOR_BGR2GRAY);
	cvtColor(next_frame, next_frame_gray, COLOR_BGR2GRAY);

	calcOpticalFlowFarneback(last_frame_gray, next_frame_gray, flow_frame, 0.4, 1, 48, 2, 8, 1.2, 0);
	for (int y = 0; y < next_frame.rows; y += 5)
	{
		for (int x = 0; x < next_frame.cols; x += 5)
		{
			const Point2f flowatxy = flow_frame.at<Point2f>(y, x) * 10;
			line(next_frame, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
			circle(next_frame, Point(x, y), 1, Scalar(0, 0, 0), -1);
		}
	}
		
	namedWindow("prew", WINDOW_AUTOSIZE);
	imshow("prew", next_frame);
	
	imwrite("000045_10_res.png", next_frame);
	

	return 0;
}