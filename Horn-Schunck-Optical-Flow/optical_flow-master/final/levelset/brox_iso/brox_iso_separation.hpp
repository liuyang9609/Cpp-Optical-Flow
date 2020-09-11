#ifndef HORNSCHUNCK_SEPARATION_HPP
#define HORNSCHUNCK_SEPARATION_HPP

#include "../../shared/types.hpp"
#include "../../shared/tensor_computation.hpp"
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "../../shared/misc.hpp"
#include "../../shared/flow_computation.hpp"
#include "../../shared/initial_separation.hpp"


void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &tp,
                          const cv::Mat_<cv::Vec6d> &tm,
                          const cv::Mat_<cv::Vec2d> &flowfield_p,
                          const cv::Mat_<cv::Vec2d> &flowfield_m,
                          cv::Mat_<cv::Vec2d> &partial_p,
                          cv::Mat_<cv::Vec2d> &partial_m,
                          const cv::Mat_<double> &data_p,
                          const cv::Mat_<double> &data_m,
                          const cv::Mat_<double> &smooth_p,
                          const cv::Mat_<double> &smooth_m,
                          const cv::Mat_<double> &phi,
                          const cv::Mat_<double> &mask,
                          std::unordered_map<std::string, parameter> &parameters,
                          double h);

void updateU(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<double> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const cv::Mat_<double> &mask,
             std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign);

void updateV(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<double> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const cv::Mat_<double> &mask,
             std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign);

void updatePhi(const cv::Mat_<double> &data_p,
               const cv::Mat_<double> &data_m,
               const cv::Mat_<double> &smooth_p,
               const cv::Mat_<double> &smooth_m,
               cv::Mat_<double> &phi,
               std::unordered_map<std::string, parameter> &parameters,
               const cv::Mat_<double> &mask,
               double h);

double H(double x);
double Hdot(double x);

#endif
