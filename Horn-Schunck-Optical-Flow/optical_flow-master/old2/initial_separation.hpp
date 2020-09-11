#ifndef INITIAL_SEPARATION_HPP
#define INITIAL_SEPARATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "types.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
#include "tensor_computation.hpp"
#include "misc.hpp"


void initial_segmentation(const cv::Mat_<cv::Vec2d> &flowfield,
                        cv::Mat_<double> &phi,
                        const std::unordered_map<std::string, parameter> &parameters,
                        cv::Vec6d &dominantmotion
                      );
void segementFlowfield(const cv::Mat_<cv::Vec2d> &f, cv::Mat_<double> &phi, const std::unordered_map<std::string, parameter> &parameters, cv::Vec6d &dominantmotion);
bool are_close_blocks(cv::Vec6d a1, cv::Vec6d a2, double Tm, double r);
void choose_better_affine(
    const cv::Mat_<bool> &merged_blocks,
    const cv::Vec6d &p_new,
    cv::Vec6d &p_old,
    const cv::Mat_<cv::Vec2d> &f,
    int blocksize
  );
double error_block(int i, int j, int blocksize, const cv::Vec6d &a_p, const cv::Mat_<cv::Vec2d> &flow);

#endif
