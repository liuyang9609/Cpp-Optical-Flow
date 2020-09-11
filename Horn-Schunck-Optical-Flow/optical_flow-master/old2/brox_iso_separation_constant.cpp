/**
  * brox et al spatial smoothness (isotropic) with segmentation 
*/

#include "brox_iso_separation.hpp"
const double DELTA = 1;
const double EPSILON_S = 0.01 * 0.01;
const double EPSILON_D = 0.01 * 0.01;
const double EPSILON_P = 0.01 * 0.01;


void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 8, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 10, 100, 10};
  parameter gamma = {"gamma", 990, 1000, 1000};
  parameter maxiter = {"maxiter", 5, 400, 1};
  parameter maxlevel = {"maxlevel", 4, 100, 1};
  parameter wrapfactor = {"wrapfactor", 95, 100, 100};
  parameter nonlinear_step = {"nonlinear_step", 3, 150, 1};
  parameter kappa = {"kappa", 100, 100, 100};
  parameter beta = {"beta", 150, 1000, 100};
  parameter deltat = {"deltat", 100, 100, 100};
  parameter phi_iter = {"phi_iter", 50, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 15, 100, 1};
  parameter Tm = {"Tm", 50, 100, 10};
  parameter Tr = {"Tr", 5, 20, 10};
  parameter Ta = {"Ta", 10, 200, 100};
  parameter blocksize = {"blocksize", 10, 100, 1};

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


void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters,
                         cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &phi, const cv::Mat_<cv::Vec2d> &initialflow, const cv::Vec6d &dominantmotion){

  cv::Mat i1smoothed, i2smoothed, i1, i2, fp_d, fm_d, seg_d;
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

  // initialize mask
  cv::Mat_<double> mask(i1smoothed.size());
  mask = 1;

  // initialization
  //flowfield = initialflow.clone();
  //flowfield_p = initialflow.clone();
  //flowfield_m = initialflow.clone();
  
  flowfield = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(0,0);
  flowfield_m = cv::Vec2d(0,0);
  
  /*flowfield = initialflow.clone();
  for (int i = 0; i < flowfield.rows; i++) {
    for (int j = 0; j < flowfield.cols; j++) {
      flowfield_p(i,j)[0] = (phi(i,j) > 0) ? initialflow(i,j)[0] : 0;
      flowfield_p(i,j)[1] = (phi(i,j) > 0) ? initialflow(i,j)[1] : 0;
      flowfield_m(i,j)[0] = (phi(i,j) < 0) ? initialflow(i,j)[0] : 0;
      flowfield_m(i,j)[1] = (phi(i,j) < 0) ? initialflow(i,j)[1] : 0;
    }
  }*/

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
    cv::resize(mask, mask, i1.size(), 0, 0, cv::INTER_NEAREST);


    // set partial flowfield to zero
    partial_p = cv::Vec2d(0,0);
    partial_m = cv::Vec2d(0,0);

    // remap image
    cv::Mat i2mapped = i2.clone();
    remap_border(i2mapped, flowfield, mask, h);

    // compute tensors
    cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2mapped, h, h) + gamma * ComputeGradientTensor(i1, i2mapped, h, h);

    // add 1px border to flowfield, parital and tensor
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_p, partial_p, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(partial_m, partial_m, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
    cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, 1);


    // main loop
    cv::Mat_<double> data_p(partial_p.size());
    cv::Mat_<double> data_m(partial_p.size());
    cv::Mat_<double> smooth_p(partial_p.size());
    cv::Mat_<double> smooth_m(partial_p.size());
    cv::Mat_<double> dataNL_p(partial_p.size());
    cv::Mat_<double> dataNL_m(partial_p.size());
    for (int i = 0; i < maxiter; i++){
      
      if (i % nonlinear_step == 0 || i == 0){
        // computed terms dont have L1 norm yet
        computeDataTerm(partial_p, t, data_p);
        computeDataTerm(partial_m, t, data_m);
        computeSmoothnessTerm(flowfield_p, partial_p, smooth_p, h, h);
        computeSmoothnessTerm(flowfield_m, partial_m, smooth_m, h, h);
        computeDataTermNL(flowfield_p+partial_p, i1, i2, dataNL_p, mask, gamma, h);
        computeDataTermNL(flowfield_m+partial_m, i1, i2, dataNL_m, mask, gamma, h);
      }

      for (int j = 0; j < iter_flow_before_phi; j++){
        /*if (j % nonlinear_step == 0 || j == 0){
          // computed terms dont have L1 norm yet
          computeDataTerm(partial_p, t, data_p);
          computeDataTerm(partial_m, t, data_m);
          computeSmoothnessTerm(flowfield_p, partial_p, smooth_p, h, h);
          computeSmoothnessTerm(flowfield_m, partial_m, smooth_m, h, h);
        }*/
        Brox_step_iso_smooth(t, flowfield_p, flowfield_m, partial_p, partial_m, data_p, data_m, smooth_p, smooth_m, phi, mask, parameters, h);
      }
      for (int k = 0; k < phi_iter; k++){
        /*if (k % nonlinear_step == 0 || k == 0){
          // computed terms dont have L1 norm yet
          computeDataTerm(partial_p, t, data_p);
          computeDataTerm(partial_m, t, data_m);
          computeSmoothnessTerm(flowfield_p, partial_p, smooth_p, h, h);
          computeSmoothnessTerm(flowfield_m, partial_m, smooth_m, h, h);
        }*/
        updatePhi(dataNL_p, dataNL_m, smooth_p, smooth_m, phi, parameters, mask, h);
        //updatePhi(data_p, data_m, smooth_p, smooth_m, phi, parameters, mask, h);
      }

      /*computeColorFlowField(partial_p, fp_d);
      cv::imshow("flowfield p", fp_d);
      computeColorFlowField(partial_m, fm_d);
      cv::imshow("flowfield m", fm_d);
      computeSegmentationImageBW(phi, i1, seg_d);
      cv::imshow("segmentation temp", seg_d);
      cv::waitKey();*/
    }

    // add partial flowfield to complete flowfield
    flowfield_p = flowfield_p + partial_p;
    flowfield_m = flowfield_m + partial_m;


    for (int i = 0; i < flowfield_p.rows; i++){
      for (int j = 0; j < flowfield_p.cols; j++){
        flowfield(i,j) = (phi(i,j) > 0) ? flowfield_p(i,j) : flowfield_m(i,j);
      }
    }
    // remove the borders
    flowfield = flowfield(cv::Rect(1, 1, i1.cols, i1.rows));
    flowfield_p = flowfield_p(cv::Rect(1, 1, i1.cols, i1.rows));
    flowfield_m = flowfield_m(cv::Rect(1, 1, i1.cols, i1.rows));
    phi = phi(cv::Rect(1, 1, i1.cols, i1.rows));
    mask = mask(cv::Rect(1, 1, i1.cols, i1.rows));
    
    
  }

  
  //flowfield = flowfield(cv::Rect(1,1,image1.cols, image1.rows));
  //phi = phi(cv::Rect(1,1,image1.cols, image1.rows));
}


/**
  * this functions performs one iteration step in the hornschunck algorithm
  * @params &cv::Mat t Brightnesstensor for computation
  * @params &cv::Mat_<cv::Vec2d> flowfield The matrix for the flowfield which is computed
  * @params &std::unordered_map<std::string, parameter> parameters The parameter hash map for the algorithm
*/
void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &t,
                          const cv::Mat_<cv::Vec2d> &flowfield_p,
                          const cv::Mat_<cv::Vec2d> &flowfield_m,
                          cv::Mat_<cv::Vec2d> &partial_p,
                          cv::Mat_<cv::Vec2d> &partial_m,
                          const cv::Mat_<double> &data_p,
                          const cv::Mat_<double> &data_m,
                          const cv::Mat_<double> &smooth_p,
                          const cv::Mat_<double> &smooth_m,
                          const cv::Mat_<double> &phi,
                          const cv::Mat_<double> &mask,
                          const std::unordered_map<std::string, parameter> &parameters,
                          double h){


  updateU(flowfield_p, partial_p, phi, data_p, smooth_p, t, mask, parameters, h, 1);
  updateU(flowfield_m, partial_m, phi, data_m, smooth_m, t, mask, parameters, h, -1);

  updateV(flowfield_p, partial_p, phi, data_p, smooth_p, t, mask, parameters, h, 1);
  updateV(flowfield_m, partial_m, phi, data_m, smooth_m, t, mask, parameters, h, -1);

}


void updateU(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<double> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const cv::Mat_<double> &mask,
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

        xm = (j > 1) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        xp = (j < p.cols - 2) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        ym = (i > 1) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        yp = (i < p.rows - 2) * (L1dot(smooth(i+1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        sum = xm + xp + ym + yp;


        // compute du
        // data terms
        tmp = mask(i,j) * H(kappa * phi(i,j) * sign) * L1dot(data(i,j), EPSILON_D) * (t(i,j)[3] * p(i,j)[1] + t(i,j)[4]);

        // smoothness terms
        tmp = tmp
              - xm * (f(i,j-1)[0] + p(i,j-1)[0])
              - xp * (f(i,j+1)[0] + p(i,j+1)[0])
              - ym * (f(i-1,j)[0] + p(i-1,j)[0])
              - yp * (f(i+1,j)[0] + p(i+1,j)[0])
              + sum * f(i,j)[0];

        // normalization
        tmp = tmp /(- mask(i,j) * H(kappa * phi(i,j) *sign) * L1dot(data(i,j), EPSILON_D) * t(i,j)[0] - sum);
        p(i,j)[0] = (1.0-omega) * p(i,j)[0] + omega * tmp;


      /*} else {
        // for now use smoothess term here

        // test for borders
        xp =  (j < p.cols-2) * 1.0/(h*h) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        xm =  (j > 1) * 1.0/(h*h) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        yp =  (i < p.rows-2) * 1.0/(h*h) * (L1dot(smooth(i+1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        ym =  (i > 1) * 1.0/(h*h) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
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
             const cv::Mat_<double> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const cv::Mat_<double> &mask,
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

         xm = (j > 1) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
         xp = (j < p.cols - 2) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
         ym = (i > 1) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
         yp = (i < p.rows - 2) * (L1dot(smooth(i+1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
         sum = xm + xp + ym + yp;


         // compute du
         // data terms
         tmp = mask(i,j) * H(kappa * phi(i,j) * sign) * L1dot(data(i,j), EPSILON_D) * (t(i,j)[3] * p(i,j)[0] + t(i,j)[5]);

         // smoothness terms
         tmp = tmp
               - xm * (f(i,j-1)[1] + p(i,j-1)[1])
               - xp * (f(i,j+1)[1] + p(i,j+1)[1])
               - ym * (f(i-1,j)[1] + p(i-1,j)[1])
               - yp * (f(i+1,j)[1] + p(i+1,j)[1])
               + sum * f(i,j)[1];

         // normalization
         tmp = tmp /(- mask(i,j) * H(kappa * phi(i,j) *sign) * L1dot(data(i,j), EPSILON_D) * t(i,j)[1] - sum);
         p(i,j)[1] = (1.0-omega) * p(i,j)[1] + omega * tmp;

      /*} else {
        // pixel lies out of the segment

        // test for borders
        xp =  (j < p.cols-2) * 1.0/(h*h) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        xm =  (j > 1) * 1.0/(h*h) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        yp =  (i < p.rows-2) * 1.0/(h*h) * (L1dot(smooth(i+1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
        ym =  (i > 1) * 1.0/(h*h) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0;
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

  //std::cout << flow_u.at<cv::Vec2d>(10,10) << ":" << flow_v.at<cv::Vec2d>(10,10) << std::endl;

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
      smooth(i,j) = tmp;
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

void computeDataTermNL(const cv::Mat_<cv::Vec2d> &f, const cv::Mat &image1, const cv::Mat &image2, cv::Mat_<double> &data, cv::Mat_<double> &mask, double gamma, double h) {
  cv::Mat i1 = image1.clone();
  cv::Mat i2 = image2.clone();
  cv::Mat i1x, i2x, i1y, i2y, kernel;

  // remove border of flowfield
  //cv::Mat_<cv::Vec2d> &f = flowfield(cv::Rect(1, 1, flowfield.cols, flowfield.rows));

  // add border to i1 and i2
  cv::copyMakeBorder(i1, i1, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(i2, i2, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);

  // remap i2
  std::cout << "nonlinear before remap" << std::endl;
  remap_border(i2, f, mask, h);
  std::cout << "nonlinear after remap" << std::endl;

  // compute fx and fy for image 1 and image2
  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(i1, i1x, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
  cv::filter2D(i2, i2x, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);

  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(i1, i1y, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
  cv::filter2D(i2, i2y, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101);
  std::cout << "nonlinear after derivates" << std::endl;

  for (int i = 0; i < f.rows; i++) {
    //std::cout << "nonlinear data computation: " << i << std::endl;
    for (int j = 0; j < f.cols; j++) {
      data(i,j) = (1-gamma) * std::pow(i2.at<double>(i,j) - i1.at<double>(i,j),2);
      data(i,j) += gamma * std::pow(i2x.at<double>(i,j) - i1x.at<double>(i,j), 2);
      data(i,j) += gamma * std::pow(i2y.at<double>(i,j) - i1y.at<double>(i,j), 2);
    }
  }
  std::cout << "nonlinear after data" << std::endl;
}


void updatePhi(const cv::Mat_<double> &data_p,
               const cv::Mat_<double> &data_m,
               const cv::Mat_<double> &smooth_p,
               const cv::Mat_<double> &smooth_m,
               cv::Mat_<double> &phi,
               const std::unordered_map<std::string, parameter> &parameters,
               const cv::Mat_<double> &mask,
               double h){

  // update the segment indicator function using implicit scheme

  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double beta = (double)parameters.at("beta").value/parameters.at("beta").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double deltat = (double)parameters.at("deltat").value/parameters.at("deltat").divfactor;

  double c1, c2, c3, c4, m, c, tmp;

  //beta = beta * std::pow(data_p.rows * data_p.cols, 0.7) * 0.0002;
  //std::cout << "beta: " << beta << std::endl;

  for (int i = 1; i < phi.rows-1; i++){
    for (int j = 1; j < phi.cols-1; j++){


      // using the vese chan discretization
      tmp = (j< phi.cols-2) * std::pow((phi(i,j+1) - phi(i,j))/h, 2) + (i>1)*(i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i-1,j))/(2*h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c1 = std::sqrt(1.0/(tmp+EPSILON_P));

      tmp = (j>1)*std::pow((phi(i,j) - phi(i,j-1))/h, 2) + (i<phi.rows-2)*(i>1)*(j>1)*std::pow((phi(i+1,j-1) - phi(i-1,j-1))/(2*h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c2 = std::sqrt(1.0/(tmp+EPSILON_P));

      tmp = (j>1)*(j<phi.cols-2)*std::pow((phi(i,j+1) - phi(i,j-1))/(2*h), 2) + (i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i,j))/(h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c3 = std::sqrt(1.0/(tmp+EPSILON_P));

      tmp = (i>1)*(j>1)*(j<phi.cols-2)*std::pow((phi(i-1,j+1) - phi(i-1,j-1))/(2*h), 2) + (i>1)*std::pow((phi(i,j) - phi(i-1,j))/(h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c4 = std::sqrt(1.0/(tmp+EPSILON_P));


      m = (deltat*Hdot(phi(i,j))*beta)/(h*h);
      c = 1+m*(c1+c2+c3+c4);
      phi(i,j) = (1.0/c)*(phi(i,j) + m*(c1*phi(i,j+1)+c2*phi(i,j-1)+c3*phi(i+1,j)+c4*phi(i-1,j))
                          -deltat*mask(i,j)*kappa*Hdot(kappa*phi(i,j))*(L1(data_p(i,j), EPSILON_D) - L1(data_m(i,j), EPSILON_D)));
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

double H(double x){
  return ( x >= 0 ) ? 1 : 0;
  //return 0.5 * (1 + (2.0/M_PI)*std::atan(x/DELTA));
}

double Hdot(double x){
  return (1.0/M_PI) * (DELTA/(DELTA*DELTA + x*x));
}
