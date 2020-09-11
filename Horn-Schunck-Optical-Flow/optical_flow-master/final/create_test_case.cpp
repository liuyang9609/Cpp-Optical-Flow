#include <iostream>
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "shared/misc.hpp"
#include "shared/flow_computation.hpp"
#include "shared/types.hpp"


void makeParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 8, 1000, 1};
  parameter omega = {"omega", 195, 200, 0.01};
  parameter sigma = {"sigma", 10, 100, 0.1};
  parameter gamma = {"gamma", 990, 1000, 0.001};
  parameter maxiter = {"maxiter", 35, 400, 1};
  parameter maxlevel = {"maxlevel", 15, 100, 1};
  parameter wrapfactor = {"wrapfactor", 99, 100, 0.01};
  parameter nonlinear_step = {"nonlinear_step", 3, 150, 1};
  parameter kappa = {"kappa", 100, 100, 0.01};
  parameter beta = {"beta", 150, 1000, 0.01};
  parameter deltat = {"deltat", 25, 100, 0.01};
  parameter phi_iter = {"phi_iter", 1, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 1, 100, 1};
  parameter Tm = {"Tm", 40, 100, 0.1};
  parameter Tr = {"Tr", 1, 20, 0.1};
  parameter Ta = {"Ta", 10, 800, 0.01};
  parameter blocksize = {"blocksize", 15, 100, 1};

  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(kappa.name, kappa));
  parameters.insert(std::make_pair<std::string, parameter>(beta.name, beta));
  parameters.insert(std::make_pair<std::string, parameter>(deltat.name, deltat));
  parameters.insert(std::make_pair<std::string, parameter>(phi_iter.name, phi_iter));
  parameters.insert(std::make_pair<std::string, parameter>(iter_flow_before_phi.name, iter_flow_before_phi));
  parameters.insert(std::make_pair<std::string, parameter>(wrapfactor.name, wrapfactor));
  parameters.insert(std::make_pair<std::string, parameter>(nonlinear_step.name, nonlinear_step));
  parameters.insert(std::make_pair<std::string, parameter>(maxlevel.name, maxlevel));
  parameters.insert(std::make_pair<std::string, parameter>(Tm.name, Tm));
  parameters.insert(std::make_pair<std::string, parameter>(Ta.name, Ta));
  parameters.insert(std::make_pair<std::string, parameter>(Tr.name, Tr));
  parameters.insert(std::make_pair<std::string, parameter>(blocksize.name, blocksize));

}

int main(int argc, char *argv[]){

  //std::string scenarioname = "yosemite";
  std::string scenarioname = "whale";
  std::unordered_map<std::string, parameter> parameters;
  makeParameters(parameters);
    
  // load images 
  cv::Mat image1 = cv::imread("images/RubberWhale/frame10.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat image2 = cv::imread("images/RubberWhale/frame11.png", CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat i1;
  cv::Mat i2;

  image1.convertTo(i1, CV_64F);
  image2.convertTo(i2, CV_64F);

  // load truth

  GroundTruth truth("images/RubberWhale/flow10.flo");

  // create scenario file
  cv::FileStorage s(scenarioname+".xml", cv::FileStorage::WRITE);

  s << "scenarioname" << scenarioname;
  s << "image1" << i1;
  s << "image2" << i2;

  s << "groundtruth" << truth.truthfield;
  s << "truthmask" << truth.mask;

  s << "interactive" << true;

  saveParameters(s, parameters);

  s.release();

}

