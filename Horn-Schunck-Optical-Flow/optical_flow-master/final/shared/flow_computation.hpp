#ifndef FLOW_COMPUTATION_HPP
#define FLOW_COMPUTATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include "types.hpp"

const double EPSILON_D = 0.01 * 0.01;
const double EPSILON_S = 0.01 * 0.01;
const double DELTA = 0.1;
const double EPSILON_P = 0.01 * 0.01;

void computeFlowField(const cv::Mat &image1,
                      const cv::Mat &image2,
                      const GroundTruth &truth,
                      cv::Mat_<double> &segmentation,
                      std::unordered_map<std::string, parameter> &parameters,
                      bool interactive,
                      cv::FileStorage &scenario
                      );

void remap_border(cv::Mat &image, const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &mask, double h);
void computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<double> &smooth, double h);
void computeDataTerm(const cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t, cv::Mat_<double> &data);
void computeDataTermNL(const cv::Mat_<cv::Vec2d> &f, const cv::Mat &image1, const cv::Mat &image2, cv::Mat_<double> &data, cv::Mat_<double> &mask, double gamma, double h) ;
double L1(double value, double epsilon);
double L1dot(double value, double epsilon);
#endif
