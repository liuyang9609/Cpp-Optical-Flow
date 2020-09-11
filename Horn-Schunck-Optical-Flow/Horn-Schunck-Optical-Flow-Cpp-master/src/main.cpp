#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/video.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "hornSchunck.cpp"
#include "plotFlow.cpp"

void preprocess(cv::Mat imagePrevRaw, cv::Mat imageNextRaw, cv::Mat &imagePrev, cv::Mat &imageNext) {
    // Converting RGB to Gray if necessary
    if (imagePrevRaw.channels() > 1) {
        cv::cvtColor(imagePrevRaw, imagePrev, cv::COLOR_BGR2GRAY);
    }
    else {
        imagePrevRaw.copyTo(imagePrev);
    }
    
    if (imageNextRaw.channels() > 1) {
        cv::cvtColor(imageNextRaw, imageNext, cv::COLOR_BGR2GRAY);
    }
    else {
        imageNextRaw.copyTo(imageNext);
    }
}

int main(int argc, char* argv[]) {
    if(argc < 6){
        std::cout << "Syntax Error - Incorrect Parameter Usage:" << std::endl;
        std::cout << "program_name ['image' (or) mp4 input path] [prev image path (or) prev frame number] [next image path (or) next frame number] [save path] [hs (or) fb]" << std::endl;
        return 0;
    }
    
    std::string inputType = argv[1];
    std::string framePrev = argv[2];
    std::string frameNext = argv[3];
    std::string savePath = argv[4];
    std::string algo = argv[5];
    
    // Read input images or video frames based on first input
    cv::Mat imagePrevRaw, imageNextRaw;
    if (inputType == "image"){
        imagePrevRaw = cv::imread(argv[2]);
        imageNextRaw = cv::imread(argv[3]);
    }
    else if (inputType.substr(inputType.length() - 4, 4) == ".mp4"){
        cv::VideoCapture capture(inputType);
        capture.set(1, stoi(framePrev));
        capture >> imagePrevRaw;
        capture.set(1, stoi(frameNext));
        capture >> imageNextRaw;
    }
    else {
        std::cout << "No image or mp4 input given! Please try again!" << std::endl;
        return 0;
    }
    // Check if files are loaded properly
    if (!(imagePrevRaw.data) || !(imageNextRaw.data)) {
        std::cout << "Can't read the images. Please check the path." << std::endl;
        return -1;
    }
    // Check if image sizes are same
    if ( imagePrevRaw.size() != imageNextRaw.size() ) {
        std::cout << "Image sizes are different. Please provide images of same size." << std::endl;
        return -1;
    }
    std::cout << "\nInput image details:" << std::endl;
    std::cout << "Raw Previous Image Size: " << imagePrevRaw.size() << std::endl;
    std::cout << "Raw Previous Image Channel Size: " << imagePrevRaw.channels() << std::endl;
    std::cout << "Raw Next Image Size: " << imageNextRaw.size() << std::endl;
    std::cout << "Raw Next Image Channel Size: " << imageNextRaw.channels() << std::endl << std::endl;
    cv::imwrite(savePath+"imagePrevRaw.png", imagePrevRaw);
    cv::imwrite(savePath+"imageNextRaw.png", imageNextRaw);
    std::cout << "Saved Raw Images in "+savePath << std::endl;

    cv::Mat imagePrev, imageNext;
    preprocess(imagePrevRaw, imageNextRaw, imagePrev, imageNext);
    std::cout << "\nAfter Preprocessing:" << std::endl;
    std::cout << "Previous Image Size: " << imagePrev.size() << std::endl;
    std::cout << "Previous Image Channel Size: " << imagePrev.channels() << std::endl;
    std::cout << "Next Image Size: " << imageNext.size() << std::endl;
    std::cout << "Next Image Channel Size: " << imageNext.channels() << std::endl << std::endl;
    
    if (algo == "hs"){
        // calculate optical flow using Horn Schunck Algorithm
        cv::Mat u, v;
        int windowSize = 5; //TBD if needed add as arguments
        int maxIterations = 100; //TBD if needed add as arguments
        double alpha = 1; //TBD if needed add as arguments
        hornSchunck hs = hornSchunck(windowSize, maxIterations, alpha);
        hs.getFlow(imagePrev, imageNext, u, v);
        cv::FileStorage file_1(savePath+"uMatrixHS.txt", cv::FileStorage::WRITE);
        file_1 << "u matrix" << u;
        cv::FileStorage file_2(savePath+"vMatrixHS.txt", cv::FileStorage::WRITE);
        file_2 << "v matrix" << v;
        plotFlow pf = plotFlow(imagePrevRaw, savePath+"hs");
        pf.plotBresenhamLine(u, v, 20, 20, 5);
        //plotLine(imagePrevRaw, u, v, 20, 20, savePath+"hs", 5);
        std::cout << "Saved HS Algorithm Results in "+savePath << std::endl;
    }
    else if (algo == "fb"){
        // Calculate optical flow using Opencv : Gunnar-Farneback Algorithm
        cv::Mat flow(imagePrev.size(), CV_32FC2);
        cv::calcOpticalFlowFarneback(imagePrev, imageNext, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cv::Mat flow_parts[2];
        split(flow, flow_parts);
        cv::FileStorage file_3(savePath+"uMatrixFB.txt", cv::FileStorage::WRITE);
        file_3 << "u matrix" << flow_parts[0];
        cv::FileStorage file_4(savePath+"vMatrixFB.txt", cv::FileStorage::WRITE);
        file_4 << "v matrix" << flow_parts[1];
        plotFlow pf = plotFlow(imagePrevRaw, savePath+"fb");
        pf.plotBresenhamLine(flow_parts[1], flow_parts[0], 20, 300, 5);
        std::cout << "Saved FB Algorithm Results in "+savePath << std::endl;
    }
    else{
        std::cout << "Currently only hs (Horn Schunck) and fb (Gunnar-Farneback) algorithm are supported!" << std::endl;
        return 0;
    }
    
    //cv::waitKey(0);
    return 0;
    
}
