#ifndef TENSOR_COMPUTATION_HPP
#define TENSOR_COMPUTATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "types.hpp"

//cv::Mat ComputeBrightnessTensor(const cv::Mat &image1, const cv::Mat &image2, double hx, double hy);
cv::Mat ComputeBrightnessTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double h);
cv::Mat ComputeGradientTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double h);


#endif
