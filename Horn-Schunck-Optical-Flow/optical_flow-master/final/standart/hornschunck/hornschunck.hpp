#ifndef HORNSCHUNCK_HPP
#define HORNSCHUNCK_HPP

#include "../../shared/types.hpp"
#include "../../shared/tensor_computation.hpp"
#include "../../shared/misc.hpp"
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <iostream>



void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t, cv::Mat_<cv::Vec2d> &flowfield, std::unordered_map<std::string, parameter> &parameters);


#endif
