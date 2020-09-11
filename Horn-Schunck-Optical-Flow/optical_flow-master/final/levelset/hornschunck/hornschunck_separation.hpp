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


void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t,
                     cv::Mat_<cv::Vec2d> &flowfield_p,
                     cv::Mat_<cv::Vec2d> &flowfield_m,
                     cv::Mat_<double> &phi,
                     std::unordered_map<std::string, parameter> &parameters,
                     double h);

void updateU(cv::Mat_<cv::Vec2d> &flowfield,
            cv::Mat_<double> &phi,
            const cv::Mat_<cv::Vec6d> &t,
            std::unordered_map<std::string, parameter> &parameters,
            double h,
            double sign);

void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               std::unordered_map<std::string, parameter> &parameters,
               double h,
               double sign);

void updatePhi(cv::Mat_<cv::Vec2d> &flowfield_p,
              cv::Mat_<cv::Vec2d> &flowfield_m,
              cv::Mat_<double> &phi,
              const cv::Mat_<cv::Vec6d> &t,
              std::unordered_map<std::string, parameter> &parameters,
              double h);

double H(double x);
double Hdot(double x);


#endif
