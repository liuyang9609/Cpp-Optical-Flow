/**
  * brox et al spatial smoothness (anisotropic) with segmentation 
*/

#include "brox_aniso_separation.hpp"
const double DELTA=1.0;
#define EPSILON 0.001


void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 10, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 15, 100, 10};
  parameter gamma = {"gamma", 500, 1000, 1000};
  parameter maxiter = {"maxiter", 40, 400, 1};
  parameter maxlevel = {"maxlevel", 4, 100, 1};
  parameter wrapfactor = {"wrapfactor", 95, 100, 100};
  parameter nonlinear_step = {"nonlinear_step", 10, 150, 1};
  parameter kappa = {"kappa", 100, 100, 100};
  parameter beta = {"beta", 4, 1000, 100};
  parameter deltat = {"deltat", 25, 100, 100};
  parameter phi_iter = {"phi_iter", 10, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 10, 100, 1};


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
}


void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters,
                         cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &phi){

  cv::Mat i1smoothed, i2smoothed, i1, i2;
  int maxlevel = parameters.at("maxlevel").value;
  int maxiter = parameters.at("maxiter").value;
  double wrapfactor = (double)parameters.at("wrapfactor").value/parameters.at("wrapfactor").divfactor;
  double gamma = (double)parameters.at("gamma").value/parameters.at("gamma").divfactor;
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  int nonlinear_step = parameters.at("nonlinear_step").value;
  int iter_flow_before_phi = parameters.at("iter_flow_before_phi").value;
  int phi_iter = parameters.at("phi_iter").value;

  double h=0;

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
  flowfield.create(i1smoothed.size());
  cv::Mat_<cv::Vec2d> partial_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> partial_m(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_m(i1smoothed.size());

  cv::Mat flowfield_wrap;
  partial_p = cv::Vec2d(0,0);
  partial_m = cv::Vec2d(0,0);
  flowfield = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(0,0);
  flowfield_m = cv::Vec2d(0,0);

  // make a 2-channel matrix with each pixel with its coordinates as value (serves as basis for flowfield remapping)
  cv::Mat remap_basis(image1.size(), CV_32FC2);
  for (int i = 0; i < image1.rows; i++){
   for (int j = 0; j < image1.cols; j++){
     remap_basis.at<cv::Vec2f>(i,j)[0] = (float)j;
     remap_basis.at<cv::Vec2f>(i,j)[1] = (float)i;
   }
  }

  // loop over levels
  for (int l = maxlevel; l >= 0; l--){
    std::cout << "Level: " << l << std::endl;

    // set steps in x and y-direction with 1.0/wrapfactor^level
    h = 1.0/std::pow(wrapfactor, l);
    std::cout << "h: " << h << std::endl;

    // scale to level, using area resampling
    cv::resize(i1smoothed, i1, cv::Size(0, 0), std::pow(wrapfactor, l), std::pow(wrapfactor, l), cv::INTER_AREA);
    cv::resize(i2smoothed, i2, cv::Size(0, 0), std::pow(wrapfactor, l), std::pow(wrapfactor, l), cv::INTER_AREA);

    // resample flowfield to current level (for now using area resampling)
    cv::resize(flowfield, flowfield, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(flowfield_p, flowfield_p, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(flowfield_m, flowfield_m, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial_p, partial_p, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(partial_m, partial_m, i1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(phi, phi, i1.size(), 0, 0, cv::INTER_AREA);


    flowfield = flowfield * wrapfactor;
    flowfield_p = flowfield_p * wrapfactor;
    flowfield_m = flowfield_m * wrapfactor;


    // set partial flowfield to zero
    partial_p = cv::Vec2d(0,0);
    partial_m = cv::Vec2d(0,0);

    // wrap image 2 with current flowfield
    flowfield.convertTo(flowfield_wrap, CV_32FC2);
    flowfield_wrap = flowfield_wrap + remap_basis(cv::Rect(0, 0, flowfield_wrap.cols, flowfield_wrap.rows));
    cv::remap(i2, i2, flowfield_wrap, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(0));

    // compute tensors
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, h, h) + gamma * ComputeGradientTensor(i1, i2, h, h);

    // add 1px border to flowfield, parital and tensor
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_p, partial_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_m, partial_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);



    // main loop
    cv::Mat_<double> data_p(partial_p.size());
    cv::Mat_<double> data_m(partial_p.size());
    cv::Mat_<cv::Vec4d> smooth_p(partial_p.size());
    cv::Mat_<cv::Vec4d> smooth_m(partial_p.size());
    for (int i = 0; i < maxiter; i++){
      for (int j = 0; j < iter_flow_before_phi; j++){
        if (j % nonlinear_step == 0 || j == 0){
          // computed terms dont have L1 norm yet
          computeDataTerm(partial_p, t, data_p);
          computeDataTerm(partial_m, t, data_m);
          computeAnisotropicSmoothnessTerm(flowfield_p, partial_p, smooth_p, h, h);
          computeAnisotropicSmoothnessTerm(flowfield_m, partial_m, smooth_m, h, h);
        }
        Brox_step_aniso_smooth(t, flowfield_p, flowfield_m, partial_p, partial_m, data_p, data_m, smooth_p, smooth_m, phi, parameters, h);
      }
      for (int k = 0; k < phi_iter; k++){
        updatePhi(data_p, data_m, smooth_p, smooth_m, phi, parameters, h);
      }
    }

    // add partial flowfield to complete flowfield
    flowfield_p = flowfield_p + partial_p;
    flowfield_m = flowfield_m + partial_m;


    for (int i = 0; i < flowfield_p.rows; i++){
      for (int j = 0; j < flowfield_p.cols; j++){
        flowfield(i,j) = (phi(i,j) > 0) ? flowfield_p(i,j) : flowfield_m(i,j);
      }
    }

  }


  //cv::Mat f_p, f_m;
  //computeColorFlowField(flowfield_p, f_p);
  //computeColorFlowField(flowfield_m, f_m);
  //cv::imshow("postive", f_p);
  //cv::imshow("negative", f_m);
  flowfield = flowfield(cv::Rect(1,1,image1.cols, image1.rows));
  phi = phi(cv::Rect(1,1,image1.cols, image1.rows));
}


/**
  * this functions performs one iteration step in the hornschunck algorithm
  * @params &cv::Mat t Brightnesstensor for computation
  * @params &cv::Mat_<cv::Vec2d> flowfield The matrix for the flowfield which is computed
  * @params &std::unordered_map<std::string, parameter> parameters The parameter hash map for the algorithm
*/
void Brox_step_aniso_smooth(const cv::Mat_<cv::Vec6d> &t,
                          const cv::Mat_<cv::Vec2d> &flowfield_p,
                          const cv::Mat_<cv::Vec2d> &flowfield_m,
                          cv::Mat_<cv::Vec2d> &partial_p,
                          cv::Mat_<cv::Vec2d> &partial_m,
                          const cv::Mat_<double> &data_p,
                          const cv::Mat_<double> &data_m,
                          const cv::Mat_<cv::Vec4d> &smooth_p,
                          const cv::Mat_<cv::Vec4d> &smooth_m,
                          const cv::Mat_<double> &phi,
                          const std::unordered_map<std::string, parameter> &parameters,
                          double h){


  updateU(flowfield_p, partial_p, phi, data_p, smooth_p, t, parameters, h, 1);
  updateU(flowfield_m, partial_m, phi, data_m, smooth_m, t, parameters, h, -1);

  updateV(flowfield_p, partial_p, phi, data_p, smooth_p, t, parameters, h, 1);
  updateV(flowfield_m, partial_m, phi, data_m, smooth_m, t, parameters, h, -1);

}


void updateU(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<cv::Vec4d> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;

  double tmp = 0;
  for (int i = 1; i < p.rows-1; i++){
    for (int j = 1; j < p.cols-1; j++){

      //if (phi(i,j)*sign > 0){
        // pixel is in the segment

        // handle borders for terms on von Neumann neighboorhood
        // isotropic part + anisotropic part

        xm = (j > 1) * (smooth(i,j-1)[0] + smooth(i,j)[0])/2.0 * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
        xp = (j < p.cols - 2) * (smooth(i,j+1)[0] + smooth(i,j)[0])/2.0 * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
        ym = (i > 1) * (smooth(i-1,j)[2] + smooth(i,j)[2])/2.0 * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
        yp = (i < p.rows - 2) * (smooth(i+1,j)[2] + smooth(i,j)[2])/2.0 * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
        sum = xm + xp + ym + yp;

        // compute du
        // data terms
        tmp = H(kappa*phi(i,j)*sign) * L1dot(data(i,j)) * (t(i,j)[3] * p(i,j)[1] + t(i,j)[4]);

        // smoothness terms (von Neumann neighboorhood)
        tmp = tmp
              - xm * (f(i,j-1)[0] + p(i,j-1)[0])
              - xp * (f(i,j+1)[0] + p(i,j+1)[0])
              - ym * (f(i-1,j)[0] + p(i-1,j)[0])
              - yp * (f(i+1,j)[0] + p(i+1,j)[0])
              + sum * f(i,j)[0];

        // remaining neighboorhoods
        tmp += (j < p.cols - 2) * ( i > 1) * (i < p.rows - 2) *
              -(alpha/(h*h)) * (smooth(i,j+1)[1] + smooth(i,j)[1])/8.0 * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0 *
              (f(i+1,j+1)[0] + p(i+1,j+1)[0] + f(i+1,j)[0] + p(i+1,j)[0] - f(i-1,j)[0] - p(i-1,j)[0] - f(i-1,j+1)[0] - p(i-1,j+1)[0]);
        tmp -= (j > 1) * (i > 1) * ( i < p.rows - 2) *
              -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i,j-1)[1])/8.0 * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0 *
              (f(i+1,j)[0] + p(i+1,j)[0] + f(i+1,j-1)[0] + p(i+1,j-1)[0] - f(i-1,j)[0] - p(i-1,j)[0] - f(i-1,j-1)[0] - p(i-1,j-1)[0]);
        tmp += (i < p.rows - 2) * (j > 1) * (j < p.cols - 2) *
              -(alpha/(h*h)) * (smooth(i+1,j)[1] + smooth(i,j)[1])/8.0 * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0 *
              (f(i+1,j+1)[0] + p(i+1,j+1)[0] + f(i,j+1)[0] + p(i,j+1)[0] - f(i,j-1)[0] - p(i,j-1)[0] - f(i+1,j-1)[0] - p(i+1,j-1)[0]);
        tmp -= (i > 1) * (j > 1) * (j < p.cols - 2) *
              -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i-1,j)[1])/8.0 * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0 *
              (f(i,j+1)[0] + p(i,j+1)[0] + f(i-1,j+1)[0] + p(i-1,j+1)[0] - f(i,j-1)[0] - p(i,j-1)[0] - f(i-1,j-1)[0] - p(i-1,j-1)[0]);

        // normalization
        tmp = tmp /(- H(kappa*phi(i,j)*sign) * L1dot(data(i,j)) * t(i,j)[0] - sum);
        
        p(i,j)[0] = (1.0-omega) * p(i,j)[0] + omega * tmp;

      /*} else {
        // for now use smoothess term here

        // test for borders
        xp =  (j < p.cols-2) * 1.0/(h*h) * (L1dot(smooth(i,j+1)) + L1dot(smooth(i,j)))/2.0;
        xm =  (j > 1) * 1.0/(h*h) * (L1dot(smooth(i,j-1)) + L1dot(smooth(i,j)))/2.0;
        yp =  (i < p.rows-2) * 1.0/(h*h) * (L1dot(smooth(i+1,j)) + L1dot(smooth(i,j)))/2.0;
        ym =  (i > 1) * 1.0/(h*h) * (L1dot(smooth(i-1,j)) + L1dot(smooth(i,j)))/2.0;
        sum = xp + xm + yp + ym;

        p(i,j)[0] = (1.0-omega) * p(i,j)[0];
        p(i,j)[0] += omega * (
          + xm * (f(i,j-1)[0] + p(i,j-1)[0])
          + xp * (f(i,j+1)[0] + p(i,j+1)[0])
          + ym * (f(i-1,j)[0] + p(i-1,j)[0])
          + yp * (f(i+1,j)[0] + p(i+1,j)[0])
          - sum * f(i,j)[0]
        )/(sum);

      }*/
    }
  }
}


void updateV(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<cv::Vec4d> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
   double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
   double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;

   double tmp = 0;
   for (int i = 1; i < p.rows-1; i++){
     for (int j = 1; j < p.cols-1; j++){

       //if (phi(i,j)*sign > 0){
         // pixel is in the segment
        
         // handle borders for terms on von Neumann neighboorhood
         // isotropic part + anisotropic part

         xm = (j > 1) * (smooth(i,j-1)[0] + smooth(i,j)[0])/2.0 * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
         xp = (j < p.cols - 2) * (smooth(i,j+1)[0] + smooth(i,j)[0])/2.0 * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
         ym = (i > 1) * (smooth(i-1,j)[2] + smooth(i,j)[2])/2.0 * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
         yp = (i < p.rows - 2) * (smooth(i+1,j)[2] + smooth(i,j)[2])/2.0 * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0 * alpha/(h*h);
         sum = xm + xp + ym + yp;


         tmp = H(kappa*phi(i,j)*sign) * L1dot(data(i,j)) * (t(i,j)[3] * p(i,j)[0] + t(i,j)[5]);

         // smoothness terms
         tmp = tmp
               - xm * (f(i,j-1)[1] + p(i,j-1)[1])
               - xp * (f(i,j+1)[1] + p(i,j+1)[1])
               - ym * (f(i-1,j)[1] + p(i-1,j)[1])
               - yp * (f(i+1,j)[1] + p(i+1,j)[1])
               + sum * f(i,j)[1];

         // remaining neighboorhoods
         tmp += (j < p.cols - 2) * ( i > 1) * (i < p.rows - 2) *
               -(alpha/(h*h)) * (smooth(i,j+1)[1] + smooth(i,j)[1])/8.0 * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0 *
               (f(i+1,j+1)[1] + p(i+1,j+1)[1] + f(i+1,j)[1] + p(i+1,j)[1] - f(i-1,j)[1] - p(i-1,j)[1] - f(i-1,j+1)[1] - p(i-1,j+1)[1]);
         tmp -= (j > 1) * (i > 1) * ( i < p.rows - 2) *
               -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i,j-1)[1])/8.0 * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0 *
               (f(i+1,j)[1] + p(i+1,j)[1] + f(i+1,j-1)[1] + p(i+1,j-1)[1] - f(i-1,j)[1] - p(i-1,j)[1] - f(i-1,j-1)[1] - p(i-1,j-1)[1]);
         tmp += (i < p.rows - 2) * (j > 1) * (j < p.cols - 2) *
               -(alpha/(h*h)) * (smooth(i+1,j)[1] + smooth(i,j)[1])/8.0 * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0 *
               (f(i+1,j+1)[1] + p(i+1,j+1)[1] + f(i,j+1)[1] + p(i,j+1)[1] - f(i,j-1)[1] - p(i,j-1)[1] - f(i+1,j-1)[1] - p(i+1,j-1)[1]);
         tmp -= (i > 1) * (j > 1) * (j < p.cols - 2) *
               -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i-1,j)[1])/8.0 * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0 *
               (f(i,j+1)[1] + p(i,j+1)[1] + f(i-1,j+1)[1] + p(i-1,j+1)[1] - f(i,j-1)[1] - p(i,j-1)[1] - f(i-1,j-1)[1] - p(i-1,j-1)[1]);

         // normalization
         tmp = tmp /(- H(kappa*phi(i,j)*sign) * L1dot(data(i,j)) * t(i,j)[1] - sum);

         p(i,j)[1] = (1.0-omega) * p(i,j)[1] + omega * tmp;

      /*} else {
        // pixel lies out of the segment

        // test for borders
        xp =  (j < p.cols-2) * 1.0/(h*h) * (L1dot(smooth(i,j+1)) + L1dot(smooth(i,j)))/2.0;
        xm =  (j > 1) * 1.0/(h*h) * (L1dot(smooth(i,j-1)) + L1dot(smooth(i,j)))/2.0;
        yp =  (i < p.rows-2) * 1.0/(h*h) * (L1dot(smooth(i+1,j)) + L1dot(smooth(i,j)))/2.0;
        ym =  (i > 1) * 1.0/(h*h) * (L1dot(smooth(i-1,j)) + L1dot(smooth(i,j)))/2.0;
        sum = xp + xm + yp + ym;

        p(i,j)[1] = (1.0-omega) * p(i,j)[1];
        p(i,j)[1] += omega * (
           + xm * (f(i,j-1)[1] + p(i,j-1)[1])
           + xp * (f(i,j+1)[1] + p(i,j+1)[1])
           + ym * (f(i-1,j)[1] + p(i-1,j)[1])
           + yp * (f(i+1,j)[1] + p(i+1,j)[1])
           - sum * f(i,j)[1]
        )/(sum);

      }*/
    }
  }
}


void computeAnisotropicSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<cv::Vec4d> &smooth, double hx, double hy){
  cv::Mat fc[2], pc[2];
  cv::Mat flow_u, flow_v, ux, uy, vx, vy, kernel, eigenvalues, eigenvectors;

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

  cv::Mat tmparray(2,2, CV_64F);
  for (int i = 0; i < p.rows; i++){
    for (int j = 0; j < p.cols; j++){
      // compute nabla(u)*nabla(u)^T + nabla(v)*nabla(v)^T
      tmparray.at<double>(0,0) = ux.at<double>(i,j) * ux.at<double>(i,j) + vx.at<double>(i,j) * vx.at<double>(i,j);
      tmparray.at<double>(1,0) = ux.at<double>(i,j) * uy.at<double>(i,j) + vx.at<double>(i,j) * vy.at<double>(i,j);
      tmparray.at<double>(0,1) = ux.at<double>(i,j) * uy.at<double>(i,j) + vx.at<double>(i,j) * vy.at<double>(i,j);
      tmparray.at<double>(1,1) = uy.at<double>(i,j) * uy.at<double>(i,j) + vy.at<double>(i,j) * vy.at<double>(i,j);
      //std::cout << tmparray << " : ";
      // compute eigenvalues
      cv::eigen(tmparray, eigenvalues, eigenvectors);

      // scale eigenvalues
      eigenvalues.at<double>(0) = L1dot(eigenvalues.at<double>(0));
      eigenvalues.at<double>(1) = L1dot(eigenvalues.at<double>(1));

      // recompute array with scaled eigenvalues
      tmparray = eigenvectors.inv() * cv::Mat::diag(eigenvalues) * eigenvectors;
      smooth(i,j)[0] = tmparray.at<double>(0,0);
      smooth(i,j)[1] = tmparray.at<double>(0,1);
      smooth(i,j)[2] = tmparray.at<double>(1,1);
      smooth(i,j)[3] = L1(eigenvalues.at<double>(0)) + L1(eigenvalues.at<double>(1));
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
      data(i,j) = tmp;
    }
  }
}



void updatePhi(const cv::Mat_<double> &data_p,
               const cv::Mat_<double> &data_m,
               const cv::Mat_<cv::Vec4d> &smooth_p,
               const cv::Mat_<cv::Vec4d> &smooth_m,
               cv::Mat_<double> &phi,
               const std::unordered_map<std::string, parameter> &parameters,
               double h){

  // update the segment indicator function using implicit scheme

  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double beta = (double)parameters.at("beta").value/parameters.at("beta").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double deltat = (double)parameters.at("deltat").value/parameters.at("deltat").divfactor;

  double c1, c2, c3, c4, m, c, tmp;

  bool stop = false;
  for (int i = 1; i < phi.rows-1; i++){
    for (int j = 1; j < phi.cols-1; j++){

      if (std::isnan(smooth_p(i,j)[3]) || std::isnan(smooth_m(i,j)[3])){
        std::cout << i << "," << j << "smooth is not a number" << std::endl;
        std::cout << smooth_p(i,j)[0] << ":" << smooth_p(i,j)[1] << ":" << smooth_p(i,j)[2] << ":" << smooth_p(i,j)[3] << std::endl;
        std::cout << smooth_m(i,j)[0] << ":" << smooth_m(i,j)[1] << ":" << smooth_m(i,j)[2] << ":" << smooth_m(i,j)[3] << std::endl;
        std::cout << phi(i,j) << " " << phi(i+1,j) << " " << phi(i-1,j) << " " << phi(i,j+1) << " " << phi(i,j-1) << std::endl;
        stop = true;
      }
      if (std::isnan(phi(i,j))){
        std::cout << i << "," << j << " phi before is not a number" << std::endl;
        stop = true;
      }

      // using the vese chan discretization
      tmp = (j< phi.cols-2) * std::pow((phi(i,j+1) - phi(i,j))/h, 2) + (i>1)*(i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i-1,j))/(2*h),2);
      c1 = (tmp == 0 || j > phi.cols-2) ? 0 : std::sqrt(1.0/tmp);

      tmp = (j>1)*std::pow((phi(i,j) - phi(i,j-1))/h, 2) + (i<phi.rows-2)*(i>1)*(j>1)*std::pow((phi(i+1,j-1) - phi(i-1,j-1))/(2*h),2);
      c2 = (tmp == 0 || j < 1) ? 0 : std::sqrt(1.0/tmp);

      tmp = (j>1)*(j<phi.cols-2)*std::pow((phi(i,j+1) - phi(i,j-1))/(2*h), 2) + (i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i,j))/(h),2);
      c3 = (tmp == 0 || i > phi.rows-2) ? 0 : std::sqrt(1.0/tmp);

      tmp = (i>1)*(j>1)*(j<phi.cols-2)*std::pow((phi(i-1,j+1) - phi(i-1,j-1))/(2*h), 2) + (i>1)*std::pow((phi(i,j) - phi(i-1,j))/(h),2);
      c4 = (tmp == 0 || i < 1) ? 0 : std::sqrt(1.0/tmp);

      m = (deltat*Hdot(phi(i,j))*beta)/(h*h);
      c = 1+m*(c1+c2+c3+c4);
      phi(i,j) = (1.0/c)*(phi(i,j) + m*(c1*phi(i,j+1)+c2*phi(i,j-1)+c3*phi(i+1,j)+c4*phi(i-1,j))
                          -deltat*kappa*Hdot(kappa*phi(i,j))*(L1(data_p(i,j)) - L1(data_m(i,j)))
                          -deltat*alpha*Hdot(phi(i,j))*(smooth_p(i,j)[3] - smooth_m(i,j)[3]));
      if (std::isnan(phi(i,j))){
        std::cout << i << "," << j << " phi is not a number" << std::endl;
        stop = true;
      }

      if (stop){
        std::exit(1);
      }
      
    }
  }

}

double L1(double value){
  return (value < 0 ) ? 0 : std::sqrt(value + EPSILON);
}


double L1dot(double value){
  value = value < 0 ? 0 : value;
  return 1.0/(2.0 * std::sqrt(value + EPSILON));
}

double H(double x){
  //return 1;
  return 0.5 * (1 + (2.0/M_PI)*std::atan(x/DELTA));
}

double Hdot(double x){
  return (1.0/M_PI) * (DELTA/(DELTA*DELTA + x*x));
}
