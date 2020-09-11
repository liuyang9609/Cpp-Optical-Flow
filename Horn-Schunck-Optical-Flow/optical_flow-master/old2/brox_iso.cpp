/**
  * Method as defined by Brox et al.

*/

#include "brox.hpp"
double const EPSILON_D = 0.001 * 0.001;
double const EPSILON_S = 0.001 * 0.001;

void debug(std::string text){
  std::cout << text << std::endl;
}


/**
  * Set up the parameters
*/
void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 80, 1000, 10};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 10, 100, 10};
  parameter gamma = {"gamma", 990, 1000, 1000};
  parameter maxiter = {"maxiter", 40, 1000, 1};
  parameter maxlevel = {"maxlevel", 4, 100, 1};
  parameter wrapfactor = {"wrapfactor", 95, 100, 100};
  parameter nonlinear_step = {"nonlinear_step", 3, 150, 1};

  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(maxlevel.name, maxlevel));
  parameters.insert(std::make_pair<std::string, parameter>(wrapfactor.name, wrapfactor));
  parameters.insert(std::make_pair<std::string, parameter>(nonlinear_step.name, nonlinear_step));

}


void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters,
                         cv::Mat_<cv::Vec2d> &flowfield){

  cv::Mat i1smoothed, i2smoothed, i1, i2;
  int maxlevel = parameters.at("maxlevel").value;
  int maxiter = parameters.at("maxiter").value;
  double wrapfactor = (double)parameters.at("wrapfactor").value/parameters.at("wrapfactor").divfactor;
  double gamma = (double)parameters.at("gamma").value/parameters.at("gamma").divfactor;
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  double hx, hy;

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
  cv::Mat flowfield_wrap;
  cv::Mat_<double> mask(i1smoothed.size());

  flowfield.create(i1smoothed.size());
  partial = cv::Vec2d(0,0);
  flowfield = cv::Vec2d(0,0);
  mask = 1;

  // make a 2-channel matrix with each pixel with its coordinates as value (serves as basis for flowfield remapping)
  cv::Mat remap_basis(image1.size(), CV_32FC2);
  for (int i = 0; i < image1.rows; i++){
    for (int j = 0; j < image1.cols; j++){
      remap_basis.at<cv::Vec2f>(i,j)[0] = (float)j;
      remap_basis.at<cv::Vec2f>(i,j)[1] = (float)i;
    }
  }

  // loop for over levels
  for (int k = maxlevel; k >= 0; k--){
    std::cout << "Level: " << k << std::endl;

    // set steps in x and y-direction with 1.0/wrapfactor^level
    hx = 1.0/std::pow(wrapfactor, k);
    hy = hx;

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, k), std::pow(wrapfactor, k), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial, partial, i1.size(), 0, 0, cv::INTER_AREA);
    //flowfield = flowfield * wrapfactor;
    
    // set partial flowfield to zero
    partial = cv::Vec2d(0,0);

    // resize mask and set it to 1
    cv::resize(mask, mask, i1.size(), 0, 0, cv::INTER_NEAREST);
    mask = 1;
    
    // add 1px border to flowfield, parital and mask
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial, partial, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, 1);

    // remap the second image with bilinear interpolation
    remap_border(i2, flowfield, mask, hx);

    // compute tensors and add 1px border
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, hy, hx) + gamma * ComputeGradientTensor(i1, i2, hx, hy);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    // main loop
    cv::Mat_<double> data(partial.size(), CV_64F);
    cv::Mat_<double> smooth(partial.size(), CV_64F);
    int nonlinear_step = parameters.at("nonlinear_step").value;
    for (int i = 0; i < maxiter; i++){
      if (i % nonlinear_step == 0 || i == 0){
        computeDataTerm(partial, t, data);
        computeSmoothnessTerm(flowfield, partial, smooth, hx, hy);
      }
      Brox_step_iso_smooth(t, flowfield, partial, data, smooth, mask, parameters, hx, hy);
    }

    // add partial flowfield to complete flowfield
    flowfield = flowfield + partial;
  }
  flowfield = flowfield(cv::Rect(1, 1, image1.cols, image1.rows));
}







void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               cv::Mat_<double> &data,
               cv::Mat_<double> &smooth,
               cv::Mat_<double> &mask,
               std::unordered_map<std::string, parameter> &parameters,
               double hx,
               double hy){

  // get parameters
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;

  // helper variables
  double xm, xp, ym, yp, sum, tmp;

  // update partial flow field
  for (int i = 1; i < p.rows - 1; i++){
    for (int j = 1; j < p.cols - 1; j++){

      // handle borders
      xm = (j > 1) * (smooth(i,j-1) + smooth(i,j))/2.0 * alpha/(hx*hx);
      xp = (j < p.cols - 2) * (smooth(i,j+1) + smooth(i,j))/2.0 * alpha/(hx*hx);
      ym = (i > 1) * (smooth(i-1,j) + smooth(i,j))/2.0 * alpha/(hy*hy);
      yp = (i < p.rows - 2) * (smooth(i+1,j) + smooth(i,j))/2.0 * alpha/(hy*hy);
      sum = xm + xp + ym + yp;



      // compute du
      // data terms
      tmp = mask(i,j) * data(i,j) * (t(i,j)[3] * p(i,j)[1] + t(i,j)[4]);

      // smoothness terms
      tmp = tmp
            - xm * (f(i,j-1)[0] + p(i,j-1)[0])
            - xp * (f(i,j+1)[0] + p(i,j+1)[0])
            - ym * (f(i-1,j)[0] + p(i-1,j)[0])
            - yp * (f(i+1,j)[0] + p(i+1,j)[0])
            + sum * f(i,j)[0];

      // normalization
      tmp = tmp /(- mask(i,j) * data(i,j) * t(i,j)[0] - sum);
      p(i,j)[0] = (1.0-omega) * p(i,j)[0] + omega * tmp;


      // same for dv
      // data terms
      tmp = mask(i,j) * data(i,j) * (t(i,j)[3] * p(i,j)[0] + t(i,j)[5]);

      // smoothness terms
      tmp = tmp
            - xm * (f(i,j-1)[1] + p(i,j-1)[1])
            - xp * (f(i,j+1)[1] + p(i,j+1)[1])
            - ym * (f(i-1,j)[1] + p(i-1,j)[1])
            - yp * (f(i+1,j)[1] + p(i+1,j)[1])
            + sum * f(i,j)[1];

      // normalization
      tmp = tmp /(- mask(i,j) * data(i,j) * t(i,j)[1] - sum);
      p(i,j)[1] = (1.0-omega) * p(i,j)[1] + omega * tmp;

    }
  }

}





void computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<double> &smooth, double hx, double hy){
  cv::Mat fc[2], pc[2];
  cv::Mat flow_u, flow_v, ux, uy, vx, vy, kernel;
  double tmp=0;

  // split flowfield in components
  cv::split(f, fc);
  flow_u = fc[0];
  flow_v = fc[1];

  // split partial flowfield in components
  cv::split(p, pc);
  flow_u = flow_u + pc[0];
  flow_v = flow_v + pc[1];

  // derivates in y-direction
  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(flow_u, uy, CV_64F, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vy, CV_64F, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  // derivates in x-dirction
  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(flow_u, ux, CV_64F, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vx, CV_64F, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  for (int i = 0; i < p.rows; i++){
    for (int j = 0; j < p.cols; j++){
      tmp = ux.at<double>(i,j) * ux.at<double>(i,j) + uy.at<double>(i,j) * uy.at<double>(i,j);
      tmp = tmp + vx.at<double>(i,j) * vx.at<double>(i,j) + vy.at<double>(i,j) * vy.at<double>(i,j);
      smooth(i,j) = L1dot(tmp, EPSILON_S);
    }
  }
}




void computeDataTerm(const cv::Mat_<cv::Vec2d> &p, const cv::Mat_<cv::Vec6d> &t, cv::Mat_<double> &data){
  double tmp;

  for (int i= 0; i < p.rows; i++){
    for (int j = 0; j < p.cols; j++){
      tmp =   t(i,j)[0] * p(i,j)[0] * p(i,j)[0]         // J11*du^2
            + t(i,j)[1] * p(i,j)[1] * p(i,j)[1]         // J22*dv^2
            + t(i,j)[2]                                 // J33
            + t(i,j)[3] * p(i,j)[0] * p(i,j)[1] * 2     // J21*du*dv
            + t(i,j)[4] * p(i,j)[0] * 2                 // J13*du
            + t(i,j)[5] * p(i,j)[1] * 2;                // J23*dv
      data(i,j) = L1dot(tmp, EPSILON_D);
    }
  }
}


double L1(double value, double epsilon){
  return (value < 0 ) ? 0 : std::sqrt(value + epsilon);
}


double L1dot(double value, double epsilon){
  value = value < 0 ? 0 : value;
  return 1.0/(2.0 * std::sqrt(value + epsilon));
}
