/**
  * perform the standard horn and schunck optical flow computation
*/

#include "hornschunck.hpp"


/**
  * this function is called at the beginning and stores the parameters in
  * an unordered hash map
  * @params &std::unordered_map parameters the hash map with the parameters
*/
void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  // we need alpha, omega, maxiter
  parameter alpha = {"alpha", 55, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter maxiter = {"maxiter", 200, 2000, 1};
  parameter sigma = {"sigma", 15, 100, 10};
  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
}


/**
  * general function which computes the flow field from two images
  * @params cv::Mat image1 first image
  * @params cv::Mat image2 second image
  * @params &std::unordered_map<std::string, paramters> parameters parameters for the flow computation
*/
cv::Mat computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters){

  // copy the images into floating point images
  cv::Mat i1, i2;
  i1 = image1.clone();
  i2 = image2.clone();
  i1.convertTo(i1, CV_64F);
  i2.convertTo(i2, CV_64F);

  std::cout << "before bluring" << std::endl;

  // smooth images
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  cv::GaussianBlur(i1, i1, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2, i2, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  cv::Mat_<cv::Vec6d> t = ComputeGradientTensor(i1, i2, 1, 1);
  cv::Mat_<cv::Vec2d> flowfield(i1.rows+2, i1.cols+2);

  // make sure all parameter exist
  if (parameters.count("maxiter") == 0 || parameters.count("alpha") == 0 || parameters.count("omega") == 0){
    std::cout << "Parameter nicht gefunden" << std::endl;
    std::exit(1);
  }

  // main loop
  for (int i = 0; i < parameters.at("maxiter").value; i++){
    HS_Stepfunction(t, flowfield, parameters);
  }

  return flowfield(cv::Rect(1,1,image1.cols, image1.rows));
}


/**
  * this functions performs one iteration step in the hornschunck algorithm
  * @params &cv::Mat t Brightnesstensor for computation
  * @params &cv::Mat_<cv::Vec2d> flowfield The matrix for the flowfield which is computed
  * @params &std::unordered_map<std::string, parameter> parameters The parameter hash map for the algorithm
*/
void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t, cv::Mat_<cv::Vec2d> &flowfield, const std::unordered_map<std::string, parameter> &parameters){

  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double h = 1;
  double a = alpha/(h*h);
  double xp, xm, yp, ym, sum;

  // maybe copyBorder flowfield -1 border into complete flowfield?
  copyMakeBorder(flowfield(cv::Rect(1,1,flowfield.cols-2, flowfield.rows-2)), flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT);

  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++ ){

      // calculate weights (it seems that borders are super important!!! why?)
      xp =  (j < flowfield.cols-2) * a;
      xm =  (j > 1) * a;
      yp =  (i < flowfield.rows-2) * a;
      ym =  (i > 1) * a;
      sum = xp + xm + yp + ym;
      //sum = ;


      // u component
      flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
      flowfield(i,j)[0] += omega * (
        - t(i-1, j-1)[4] - t(i-1, j-1)[3] * flowfield(i,j)[1]
        + yp * flowfield(i+1,j)[0]
        + ym * flowfield(i-1,j)[0]
        + xp * flowfield(i,j+1)[0]
        + xm * flowfield(i,j-1)[0]
      )/(t(i-1, j-1)[0] + sum);

      // v component
      flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
      flowfield(i,j)[1] += omega * (
        - t(i-1, j-1)[5] - t(i-1, j-1)[3] * flowfield(i,j)[0]
        + yp * flowfield(i+1,j)[1]
        + ym * flowfield(i-1,j)[1]
        + xp * flowfield(i,j+1)[1]
        + xm * flowfield(i,j-1)[1]
      )/(t(i-1, j-1)[1] + sum);
    }
  }
}
