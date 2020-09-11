#ifndef HORNSCHUNCK_SEPARATION_HPP
#define HORNSCHUNCK_SEPARATION_HPP

#include "types.hpp"
#include "tensor_computation.hpp"
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "misc.hpp"


void setupParameters(std::unordered_map<std::string, parameter> &parameters);
void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters, cv::Mat_<cv::Vec2d> &flowfield);
void Brox_step_aniso_smooth(const cv::Mat_<cv::Vec6d> &t,
                          const cv::Mat_<cv::Vec2d> &flowfield_p,
                          const cv::Mat_<cv::Vec2d> &flowfield_m,
                          cv::Mat_<cv::Vec2d> &partial_p,
                          cv::Mat_<cv::Vec2d> &partial_m,
                          const cv::Mat_<double> &data_p,
                          const cv::Mat_<double> &data_m,
                          const cv::Mat_<cv::Vec4d> &smooth_p,
                          const cv::Mat_<cv::Vec4d> &smooth_m,
                          const cv::Mat_<double> &phi,
                          const std::unordered_map<std::string, parameter> &parameters,
                          double h);

void updateU(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<cv::Vec4d> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign);

void updateV(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<cv::Vec4d> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign);

void updatePhi(const cv::Mat_<double> &data_p,
               const cv::Mat_<double> &data_m,
               const cv::Mat_<cv::Vec4d> &smooth_p,
               const cv::Mat_<cv::Vec4d> &smooth_m,
               cv::Mat_<double> &phi,
               const std::unordered_map<std::string, parameter> &parameters,
               double h);

void computeAnisotropicSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<cv::Vec4d> &smooth, double hx, double hy);
void computeDataTerm(const cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t, cv::Mat_<double> &data);
double H(double x);
double Hdot(double x);
double L1(double value);
double L1dot(double value);


#endif
