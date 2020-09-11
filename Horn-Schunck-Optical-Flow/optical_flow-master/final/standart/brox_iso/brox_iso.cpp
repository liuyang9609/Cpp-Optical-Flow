/**
  * Method as defined by Brox et al.

*/

#include "brox_iso.hpp"
void debug(std::string text){
  std::cout << text << std::endl;
}


void computeFlowField(const cv::Mat &image1,
                      const cv::Mat &image2,
                      const GroundTruth &truth,
                      cv::Mat_<double> &segmentation,
                      std::unordered_map<std::string, parameter> &parameters,
                      bool interactive,
                      cv::FileStorage &scenario
                      ) {

  cv::Mat i1smoothed, i2smoothed, i1, i2, i2mapped;
  int maxlevel = parameters.at("maxlevel").value;
  int maxiter = parameters.at("maxiter").value;
  double wrapfactor = getParameter("wrapfactor", parameters);
  double gamma = getParameter("gamma", parameters);
  double sigma = getParameter("sigma", parameters);
  double h;

  // make deepcopy, so images are untouched
  i1smoothed = image1.clone();
  i2smoothed = image2.clone();

  // convert to floating point images
  i1smoothed.convertTo(i1smoothed, CV_64F);
  i2smoothed.convertTo(i2smoothed, CV_64F);

  // blurring of the images (before resizing)
  cv::GaussianBlur(i1smoothed, i1smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2smoothed, i2smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // initialize parital and complete flowfield
  cv::Mat_<cv::Vec2d> partial(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield(i1smoothed.size());
  cv::Mat_<double> mask(i1smoothed.size());

  partial = cv::Vec2d(0,0);
  flowfield = cv::Vec2d(0,0);
  mask = 1;

  // loop for over levels
  for (int k = maxlevel; k >= 0; k--){
    std::cout << "Level: " << k << std::endl;

    // set steps in x and y-direction with 1.0/wrapfactor^level
    h = 1.0/std::pow(wrapfactor, k);

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial, partial, i1.size(), 0, 0, cv::INTER_AREA);
    
    // set partial flowfield to zero
    partial = cv::Vec2d(0,0);

    // resize mask and set it to 1
    cv::resize(mask, mask, i1.size(), 0, 0, cv::INTER_NEAREST);
    mask = 1;

    // remap the second image with bilinear interpolation
    i2mapped = i2.clone();
    remap_border(i2mapped, flowfield, mask, h);
    
    // add 1px border to flowfield, parital and mask
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial, partial, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, 1);

    // compute tensors and add 1px border
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2mapped, h) + gamma * ComputeGradientTensor(i1, i2mapped, h);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    // main loop
    cv::Mat_<double> data(partial.size(), CV_64F);
    cv::Mat_<double> smooth(partial.size(), CV_64F);
    int nonlinear_step = parameters.at("nonlinear_step").value;
    
    for (int i = 0; i < maxiter; i++){
      if (i % nonlinear_step == 0 || i == 0){
        computeDataTerm(partial, t, data);
        computeSmoothnessTerm(flowfield, partial, smooth, h);
      }
      Brox_step_iso_smooth(t, flowfield, partial, data, smooth, mask, parameters, h);
    }

    // add partial flowfield to complete flowfield
    flowfield = flowfield + partial;

    // remove borders
    flowfield = flowfield(cv::Rect(1, 1, i1.cols, i1.rows));
    partial = partial(cv::Rect(1, 1, i1.cols, i1.rows));
    mask = mask(cv::Rect(1, 1, i1.cols, i1.rows));
  }

  if (interactive) {
    displayFlow("flow", flowfield);
    displayError("error", flowfield, truth);
    
    if (truth.isSet) {
      std::cout << "AAE:" << truth.computeAngularError(flowfield) << std::endl;
      std::cout << "EPE:" << truth.computeEndpointError(flowfield) << std::endl;
    }

    std::string scenarioname;
    scenario["scenarioname"] >> scenarioname;
    cv::FileStorage s(scenarioname+".xml", cv::FileStorage::APPEND);
    s << "initialflow" << flowfield;
    s.release();
  }

}







void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
                         const cv::Mat_<cv::Vec2d> &f,
                         cv::Mat_<cv::Vec2d> &p,
                         cv::Mat_<double> &data,
                         cv::Mat_<double> &smooth,
                         cv::Mat_<double> &mask,
                         std::unordered_map<std::string, parameter> &parameters,
                         double h
                         ){

  // get parameters
  double alpha = getParameter("alpha", parameters);
  double omega = getParameter("omega", parameters);

  // helper variables
  double xm, xp, ym, yp, sum, tmp;

  // update partial flow field
  for (int i = 1; i < p.rows - 1; i++){
    for (int j = 1; j < p.cols - 1; j++){

      // handle borders
      xm = (j > 1) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
      xp = (j < p.cols - 2) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
      ym = (i > 1) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
      yp = (i < p.rows - 2) * (L1dot(smooth(i+1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
      sum = xm + xp + ym + yp;



      // compute du
      // data terms
      tmp = mask(i,j) * L1dot(data(i,j), EPSILON_D) * (t(i,j)[3] * p(i,j)[1] + t(i,j)[4]);

      // smoothness terms
      tmp = tmp
            - xm * (f(i,j-1)[0] + p(i,j-1)[0])
            - xp * (f(i,j+1)[0] + p(i,j+1)[0])
            - ym * (f(i-1,j)[0] + p(i-1,j)[0])
            - yp * (f(i+1,j)[0] + p(i+1,j)[0])
            + sum * f(i,j)[0];

      // normalization
      tmp = tmp /(- mask(i,j) * L1dot(data(i,j), EPSILON_D) * t(i,j)[0] - sum);
      p(i,j)[0] = (1.0-omega) * p(i,j)[0] + omega * tmp;


      // same for dv
      // data terms
      tmp = mask(i,j) * L1dot(data(i,j), EPSILON_D) * (t(i,j)[3] * p(i,j)[0] + t(i,j)[5]);

      // smoothness terms
      tmp = tmp
            - xm * (f(i,j-1)[1] + p(i,j-1)[1])
            - xp * (f(i,j+1)[1] + p(i,j+1)[1])
            - ym * (f(i-1,j)[1] + p(i-1,j)[1])
            - yp * (f(i+1,j)[1] + p(i+1,j)[1])
            + sum * f(i,j)[1];

      // normalization
      tmp = tmp /(- mask(i,j) * L1dot(data(i,j), EPSILON_D) * t(i,j)[1] - sum);
      p(i,j)[1] = (1.0-omega) * p(i,j)[1] + omega * tmp;

    }
  }

}

