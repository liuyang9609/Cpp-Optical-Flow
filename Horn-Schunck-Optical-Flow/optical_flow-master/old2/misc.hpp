#ifndef MISC_HPP
#define MISC_HPP

#include "types.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <iostream>

void computeColorFlowField(const cv::Mat_<cv::Vec2d> &f, cv::Mat &img);
void computeColorFlowFieldError(const cv::Mat_<cv::Vec2d> &f, GroundTruth &truth, cv::Mat &img);
void loadBarronFile(std::string filename, cv::Mat_<cv::Vec2d> &truth);
void TrackbarCallback(int value, void *userdata);
void computeColorFlowField2(const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat &img);
double CalcAngularError(const cv::Mat_<cv::Vec2d> &flowfield, const cv::Mat_<cv::Vec2d> truth);

void setupParameters(std::unordered_map<std::string, parameter> &parameters);
void computeFlowField(const cv::Mat &image1,
                 const cv::Mat &image2,
                 std::unordered_map<std::string, parameter> &parameters,
                 cv::Mat_<cv::Vec2d> &flowfield);
void computeFlowField(const cv::Mat &image1,
                 const cv::Mat &image2,
                 std::unordered_map<std::string, parameter> &parameters,
                 cv::Mat_<cv::Vec2d> &flowfield,
                 cv::Mat_<double> &phi,
                 const cv::Mat_<cv::Vec2d> &initialflow,
                 const cv::Vec6d &dominantmotion);

void remap_border(cv::Mat &image, const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &mask, double h);
void computeSegmentationImage(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation);
void computeSegmentationImageBW(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation);
#endif
