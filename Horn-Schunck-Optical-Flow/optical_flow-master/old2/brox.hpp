#ifndef BROX_HPP
#define BROX_HPP

#include "types.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include "tensor_computation.hpp"
#include "misc.hpp"

void Brox_step_aniso_smooth(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               cv::Mat_<double> &data,
               cv::Mat_<cv::Vec3d> &smooth,
               cv::Mat_<double> &mask,
               std::unordered_map<std::string, parameter> &parameters,
               double hx,
               double hy);

void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               cv::Mat_<double> &data,
               cv::Mat_<double> &smooth,
               cv::Mat_<double> &mask,
               std::unordered_map<std::string, parameter> &parameters,
               double hx,
               double hy);


void computeAnisotropicSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<cv::Vec3d> &smooth, double hx, double hy);
void computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<double> &smooth, double hx, double hy);
void computeDataTerm(const cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t, cv::Mat_<double> &data);
double L1(double value, double epsilon);
double L1dot(double value, double epsilon);

#endif
