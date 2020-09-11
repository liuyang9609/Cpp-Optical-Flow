#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

class plotFlow{
public:
    cv::Mat image;
    std::string savePath;
    
    plotFlow(cv::Mat inpImage, std::string inpSavePath){
        image = inpImage;
        savePath = inpSavePath;
    }
    
    int Sign(int x){
        if(x<0) return -1;
        else if(x>0) return 1;
        else return 0;
    }
    
    void setPixelColor(cv::Mat &image, int x, int y, int r, int g, int b){
        if ((x<image.rows-1) & (x >= 0)){
            if ((y<image.cols-1) & (y >= 0)){
                image.at<cv::Vec3b>(x, y)[0] = r;
                image.at<cv::Vec3b>(x, y)[1] = g;
                image.at<cv::Vec3b>(x, y)[2] = b;
            }
        }
    }
    
    void moveLateral(int &x, int &y, double &R, int signX, int signY, int distanceX, int distanceY){
        x+= signX; R+= distanceY;
            if (R >= distanceX)
            {
                y+= signY;
                R-= distanceX;
            }
    }
    
    void bresenhamPoints(cv::Mat &image, int xStart, int yStart, int xEnd, int yEnd, int r, int g, int b){
        int distanceX = xEnd - xStart;
        int distanceY = yEnd - yStart;
        int signX = Sign(distanceX);
        int signY = Sign(distanceY);
        distanceX = abs(distanceX);
        distanceY = abs(distanceY);
        int distance = std::max(distanceX, distanceY);
        double R = distance / 2;
        int x = xStart;
        int y = yStart;
        if(distanceX > distanceY){
            for(int i=0; i<distance; i++){
                setPixelColor(image, x, y, r, g, b);
                moveLateral(x, y, R, signX, signY, distanceX, distanceY);
            }
        }
        else{
            for(int i=0; i<distance; i++){
                setPixelColor(image, x, y, r, g, b);
                moveLateral(y, x, R, signY, signX, distanceY, distanceX);
            }
        }
    }
    
    void plotBresenhamLine(cv::Mat u, cv::Mat v, int delta, float scale, int outlier) {
        cv::Mat imagePlot = image.clone();
        for (int x1 = 0; x1 < image.rows; x1 += delta){
            for (int y1 = 0; y1 < image.cols; y1 += delta) {
                int x2 = (int)(x1 + (u.at<double>(x1, y1)*scale));
                int y2 = (int)(y1 + (v.at<double>(x1, y1)*scale));
                if (outlier > 0) {
                    if ((u.at<double>(x1, y1) < outlier) & (v.at<double>(x1, y1) < outlier) & (u.at<double>(x1, y1) > -1*outlier) & (v.at<double>(x1, y1) > -1*outlier)){
                        bresenhamPoints(imagePlot, x1, y1, x2, y2, 0, 255, 0);
                    }
                }
                else{
                    bresenhamPoints(imagePlot, x1, y1, x2, y2, 0, 255, 0);
                }
                setPixelColor(imagePlot, x2, y2, 0, 0, 255); // change end point color to red
            }
        }
        cv::namedWindow("bresenhamLineFlow image", cv::WINDOW_AUTOSIZE);
        cv::imshow("bresenhamLineFlow image", imagePlot);
        cv::imwrite(savePath+"bresenhamLineFlow.png", imagePlot);
    }
};
