#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

class hornSchunck{
public:
    int windowSize, maxIterations;
    double alpha;
    
    hornSchunck(int inpWindowSize, int inpMaxIterations, double inpAlpha){
        windowSize = inpWindowSize;
        maxIterations = inpMaxIterations;
        alpha = inpAlpha;
    }
    
    void getGradients(cv::Mat imagePrev, cv::Mat imageNext, cv::Mat &gradX, cv::Mat &gradY, cv::Mat &gradT){
        cv::Mat imagePrevNorm, imageNextNorm;
        
        // Convert image values to 64 bit float single channel
        imagePrev.convertTo(imagePrevNorm, CV_64FC1);
        imageNext.convertTo(imageNextNorm, CV_64FC1);
        
        // Obtain gradient in X direction and Y direction using a 3x3 Sobel Filter
        Sobel(imagePrevNorm, gradX, -1, 1, 0, 3);
        Sobel(imagePrevNorm, gradY, -1, 0, 1, 3);
        
        //cv::Mat gradX1, gradX2, gradY1, gradY2;
        //Sobel(imagePrevNorm, gradX1, -1, 1, 0, 3);
        //Sobel(imageNextNorm, gradX2, -1, 1, 0, 3);
        //Sobel(imagePrevNorm, gradY1, -1, 0, 1, 3);
        //Sobel(imageNextNorm, gradY2, -1, 0, 1, 3);
        //gradX = (gradX1+gradX2)/2.0;
        //gradY = (gradY1+gradY2)/2.0;
        
        // Obtain gradient in T direction by subtracting both the images
        gradT = imageNextNorm - imagePrevNorm;
        
    }
    
    void getFlow(cv::Mat imagePrev, cv::Mat imageNext, cv::Mat &u, cv::Mat &v){
        // Get gradients
        cv::Mat gradX, gradY, gradT;
        getGradients(imagePrev, imageNext, gradX, gradY, gradT);
        
        // Initialize u and v matrices with zeros of same size and format as gradT
        u = cv::Mat::zeros(gradT.rows, gradT.cols, CV_64FC1);
        v = cv::Mat::zeros(gradT.rows, gradT.cols, CV_64FC1);
        
        // Get kernel and anchor for averaging u and v matrices
        cv::Mat kernel = cv::Mat::ones(windowSize, windowSize, CV_64FC1) / pow(windowSize, 2);
        cv::Point anchor(windowSize-(windowSize/2)-1,windowSize-(windowSize/2)-1);
        
        for (int i = 0; i < maxIterations; i++) {
            cv::Mat uAvg, vAvg, gradXuAvg, gradYvAvg, gradXgradX, gradYgradY, updateConst, uUpdateConst, vUpdateConst;
            
            // Convolving image with a kernel, low pass filtering
            filter2D(u, uAvg, u.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
            filter2D(v, vAvg, v.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);
            
            multiply(gradX, uAvg, gradXuAvg);
            multiply(gradY, vAvg, gradYvAvg);
            multiply(gradX, gradX, gradXgradX);
            multiply(gradY, gradY, gradYgradY);
            
            divide((gradXuAvg + gradYvAvg + gradT), (pow(alpha,2) + gradXgradX +gradYgradY), updateConst);
            multiply(gradX, updateConst, uUpdateConst);
            multiply(gradY, updateConst, vUpdateConst);
            
            u = uAvg - uUpdateConst;
            v = vAvg - vUpdateConst;
        }
    }
};
