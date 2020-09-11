#ifndef BROX_ISO_HPP
#define BROX_ISO_HPP

#include "../../shared/types.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include "../../shared/tensor_computation.hpp"
#include "../../shared/misc.hpp"
#include "../../shared/flow_computation.hpp"

void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               cv::Mat_<double> &data,
               cv::Mat_<double> &smooth,
               cv::Mat_<double> &mask,
               std::unordered_map<std::string, parameter> &parameters,
               double h);

#endif
