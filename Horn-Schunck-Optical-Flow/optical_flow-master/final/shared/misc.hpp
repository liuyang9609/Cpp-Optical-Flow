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
void computeColorFlowFieldError(const cv::Mat_<cv::Vec2d> &f, const GroundTruth &truth, cv::Mat &img);
void loadBarronFile(std::string filename, cv::Mat_<cv::Vec2d> &truth);
void TrackbarCallback(int value, void *userdata);
void computeColorFlowField2(const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat &img);
double CalcAngularError(const cv::Mat_<cv::Vec2d> &flowfield, const cv::Mat_<cv::Vec2d> truth);

void loadParameters(cv::FileNode &node, std::unordered_map<std::string, parameter> &parameters);
void saveParameters(cv::FileStorage &storage, std::unordered_map<std::string, parameter> &parameters);
double getParameter(std::string name, std::unordered_map<std::string, parameter> &parameters);
void computeSegmentationImage(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation);
void computeSegmentationImageBW(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation);

void displayFlow(std::string windowname, const cv::Mat_<cv::Vec2d> &flowfield);
void displayError(std::string windowname, const cv::Mat_<cv::Vec2d> &flowfield, const GroundTruth &truth);
void displayImage(std::string windowname, const cv::Mat &image);
void displaySegmentation(std::string windowname, const cv::Mat_<double> &phi);
void displaySegmentationBW(std::string windowname, const cv::Mat_<double> &phi, const cv::Mat &image);
#endif
