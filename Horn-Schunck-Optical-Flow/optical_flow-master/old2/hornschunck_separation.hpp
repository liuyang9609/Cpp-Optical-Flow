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
void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t,
                     cv::Mat_<cv::Vec2d> &flowfield_p,
                     cv::Mat_<cv::Vec2d> &flowfield_m,
                     cv::Mat_<double> &phi,
                     const std::unordered_map<std::string, parameter> &parameters);

void updateU(cv::Mat_<cv::Vec2d> &flowfield,
            cv::Mat_<double> &phi,
            const cv::Mat_<cv::Vec6d> &t,
            const std::unordered_map<std::string, parameter> &parameters,
            double h,
            double sign);

void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h,
               double sign);

void updatePhi(cv::Mat_<cv::Vec2d> &flowfield_p,
              cv::Mat_<cv::Vec2d> &flowfield_m,
              cv::Mat_<double> &phi,
              const cv::Mat_<cv::Vec6d> &t,
              const std::unordered_map<std::string, parameter> &parameters,
              double h);

double H(double x);
double Hdot(double x);


#endif
