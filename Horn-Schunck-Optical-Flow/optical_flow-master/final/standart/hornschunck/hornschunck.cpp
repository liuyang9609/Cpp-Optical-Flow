/**
  * Horn Schunck method with additional gradient constraint
*/

#include "hornschunck.hpp"


void computeFlowField(const cv::Mat &image1,
                      const cv::Mat &image2,
                      const GroundTruth &truth,
                      cv::Mat_<double> &segmentation,
                      std::unordered_map<std::string, parameter> &parameters,
                      bool interactive,
                      cv::FileStorage &scenario
                      ) {


  std::cout << "started computation" << std::endl;

  // convert images into 64 bit floating point images
  cv::Mat i1, i2;
  i1 = image1.clone();
  i1.convertTo(i1, CV_64F);
  i2 = image2.clone();
  i2.convertTo(i2, CV_64F);

  displayImage("image1", i1);

  // parameters
  double sigma = getParameter("sigma", parameters);
  double gamma = getParameter("gamma", parameters);
  double maxiter = getParameter("maxiter", parameters);

  // Blurring
  cv::GaussianBlur(i1, i1, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2, i2, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // compute Brightness and Gradient Tensors
  cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, 1) + gamma * ComputeGradientTensor(i1, i2, 1);

  // create flowfield
  cv::Mat_<cv::Vec2d> flowfield(i1.size());
  flowfield = cv::Vec2d(0,0);

  cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1 , 1, cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(t, t, 1, 1, 1 , 1, cv::BORDER_REPLICATE, 0);

  // main loop
  for (int i = 0; i < maxiter; i++){
    HS_Stepfunction(t, flowfield, parameters);
  }

  flowfield = flowfield(cv::Rect(1,1,image1.cols, image1.rows));
  
  if (interactive) {
    displayFlow("flow", flowfield);
    displayError("error", flowfield, truth);
    
    if (truth.isSet) {
      std::cout << "AAE:" << truth.computeAngularError(flowfield) << std::endl;
      std::cout << "EPE:" << truth.computeEndpointError(flowfield) << std::endl;
    }
  }
}





/**
  * this functions performs one iteration step in the hornschunck algorithm
  * @params &cv::Mat t Brightnesstensor for computation
  * @params &cv::Mat_<cv::Vec2d> flowfield The matrix for the flowfield which is computed
  * @params &std::unordered_map<std::string, parameter> parameters The parameter hash map for the algorithm
*/
void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t, cv::Mat_<cv::Vec2d> &flowfield, std::unordered_map<std::string, parameter> &parameters){

  double omega = getParameter("omega", parameters);
  double alpha = getParameter("alpha", parameters);
  double h = 1;
  double a = alpha/(h*h);
  double xp, xm, yp, ym, sum;

  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++ ){

      // calculate weights (it seems that borders are super important!!! why?)
      xp =  (j < flowfield.cols-2) * a;
      xm =  (j > 1) * a;
      yp =  (i < flowfield.rows-2) * a;
      ym =  (i > 1) * a;
      sum = xp + xm + yp + ym;

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
