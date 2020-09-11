/**
  * brox et al spatial smoothness (isotropic) with segmentation 
*/

#include "../brox_iso_separation.hpp"

void computeFlowField(const cv::Mat &image1,
                      const cv::Mat &image2,
                      const GroundTruth &truth,
                      cv::Mat_<double> &segmentation,
                      std::unordered_map<std::string, parameter> &parameters,
                      bool interactive,
                      cv::FileStorage &scenario
                      ){

  cv::Mat i1smoothed, i2smoothed, i1, i2, i2p, i2m, fp_d, fm_d, seg_d;
  int maxlevel = parameters.at("maxlevel").value;
  int maxiter = parameters.at("maxiter").value;
  int nonlinear_step = parameters.at("nonlinear_step").value;
  int iter_flow_before_phi = parameters.at("iter_flow_before_phi").value;
  int phi_iter = parameters.at("phi_iter").value;
  double wrapfactor = getParameter("wrapfactor", parameters);
  double gamma = getParameter("gamma", parameters);
  double sigma = getParameter("sigma", parameters);
  double h = 0;

  // make deepcopy, so images are untouched
  i1smoothed = image1.clone();
  i2smoothed = image2.clone();

  // convert to floating point images
  i1smoothed.convertTo(i1smoothed, CV_64F);
  i2smoothed.convertTo(i2smoothed, CV_64F);

  // blurring of the images (before resizing)
  cv::GaussianBlur(i1smoothed, i1smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2smoothed, i2smoothed, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // create partial and complete flowfield
  cv::Mat_<cv::Vec2d> flowfield(i1smoothed.size());
  cv::Mat_<cv::Vec2d> partial_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> partial_m(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_p(i1smoothed.size());
  cv::Mat_<cv::Vec2d> flowfield_m(i1smoothed.size());
  cv::Mat_<double> phi(i1smoothed.size());

  // initialize mask
  cv::Mat_<double> mask(i1smoothed.size());
  mask = 1;

  // initialflow
  cv::Mat_<cv::Vec2d> initialflow;
  scenario["initialflow"] >> initialflow;
  
  // initial segmentation
  cv::Vec6d dominantmotion;
  if (segmentation.empty()) {
    segmentation.create(i1smoothed.size());
    segmentation = 1;
  }

  if (interactive) {
    char keyCode = 'y';
    while (keyCode == 'y'){
      displaySegmentation("initialsegmentation", segmentation);
      std::cout << "new separation?" << std::endl;
      keyCode = cv::waitKey();
      if (keyCode == 'y') {
        initial_segmentation(initialflow, segmentation, parameters, dominantmotion);
      }
    }
  } else {
    initial_segmentation(initialflow, segmentation, parameters, dominantmotion);
  }

  phi = segmentation.clone();


  flowfield = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(0,0);
  flowfield_m = cv::Vec2d(0,0);


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
    i2p = i2.clone();
    i2m = i2.clone();
    remap_border(i2p, flowfield_p, mask, h);
    remap_border(i2m, flowfield_m, mask, h);

    // compute tensors
    cv::Mat_<cv::Vec6d> tp = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2p, h) + gamma * ComputeGradientTensor(i1, i2p, h);
    cv::Mat_<cv::Vec6d> tm = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2m, h) + gamma * ComputeGradientTensor(i1, i2m, h);
    
    // add 1px border to flowfield, parital and tensor
    cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(partial_p, partial_p, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(partial_m, partial_m, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(tp, tp, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(tm, tm, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 0);
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED, 1);


    // main loop
    cv::Mat_<double> data_p(partial_p.size());
    cv::Mat_<double> data_m(partial_p.size());
    cv::Mat_<double> smooth_p(partial_p.size());
    cv::Mat_<double> smooth_m(partial_p.size());
    cv::Mat_<double> data_pNL(partial_p.size());
    cv::Mat_<double> data_mNL(partial_p.size());
    for (int i = 0; i < maxiter; i++){
        
      if (i % nonlinear_step == 0 || i == 0){
        computeDataTerm(partial_p, tp, data_p);
        computeDataTerm(partial_m, tm, data_m);
        //computeDataTermNL(flowfield_p+partial_p, i1, i2, data_pNL, mask, gamma, h);
        //computeDataTermNL(flowfield_m+partial_m, i1, i2, data_mNL, mask, gamma, h);
        computeSmoothnessTerm(flowfield_p, partial_p, smooth_p, h);
        computeSmoothnessTerm(flowfield_m, partial_m, smooth_m, h);
      }
      
      for (int j = 0; j < iter_flow_before_phi; j++){
        Brox_step_iso_smooth(tp, tm, flowfield_p, flowfield_m, partial_p, partial_m, data_p, data_m, smooth_p, smooth_m, phi, mask, parameters, h);
      }
      
      for (int k = 0; k < phi_iter; k++){
        updatePhi(data_p, data_m, smooth_p, smooth_m, phi, parameters, mask, h);
      }
      
    }

    // add partial flowfield to complete flowfield
    flowfield_p = flowfield_p + partial_p;
    flowfield_m = flowfield_m + partial_m;

    // remove the borders
    flowfield = flowfield(cv::Rect(1, 1, i1.cols, i1.rows));
    flowfield_p = flowfield_p(cv::Rect(1, 1, i1.cols, i1.rows));
    flowfield_m = flowfield_m(cv::Rect(1, 1, i1.cols, i1.rows));
    phi = phi(cv::Rect(1, 1, i1.cols, i1.rows));
    mask = mask(cv::Rect(1, 1, i1.cols, i1.rows));
    if (interactive) {
      displayFlow("flowfield p", flowfield_p);
      displayFlow("flowfield m", flowfield_m);
      displaySegmentation("phi temp", phi);
      displaySegmentationBW("segmentation temp", phi, i1);
      displaySegmentationBW("mask temp", mask, i1);
      cv::waitKey();
    }
  }
  
  for (int i = 0; i < flowfield_p.rows; i++){
    for (int j = 0; j < flowfield_p.cols; j++){
      flowfield(i,j) = (phi(i,j) > 0) ? flowfield_p(i,j) : flowfield_m(i,j);
    }
  }

  if (interactive) {
    displayFlow("flow", flowfield);
    displayError("error", flowfield, truth);
    displaySegmentation("finalsegmentation", phi);
     
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
void Brox_step_iso_smooth(const cv::Mat_<cv::Vec6d> &tp,
                          const cv::Mat_<cv::Vec6d> &tm,
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
                          std::unordered_map<std::string, parameter> &parameters,
                          double h){


  updateU(flowfield_p, partial_p, phi, data_p, smooth_p, tp, mask, parameters, h, 1);
  updateU(flowfield_m, partial_m, phi, data_m, smooth_m, tm, mask, parameters, h, -1);

  updateV(flowfield_p, partial_p, phi, data_p, smooth_p, tp, mask, parameters, h, 1);
  updateV(flowfield_m, partial_m, phi, data_m, smooth_m, tm, mask, parameters, h, -1);

}


void updateU(const cv::Mat_<cv::Vec2d> &f,
             cv::Mat_<cv::Vec2d> &p,
             const cv::Mat_<double> &phi,
             const cv::Mat_<double> data,
             const cv::Mat_<double> smooth,
             const cv::Mat_<cv::Vec6d> &t,
             const cv::Mat_<double> &mask,
             std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = getParameter("alpha", parameters);
  double kappa = getParameter("kappa", parameters);
  double omega = getParameter("omega", parameters);

  double tmp = 0;
  for (int i = 1; i < p.rows-1; i++){
    for (int j = 1; j < p.cols-1; j++){

      //if (phi(i,j)*sign > 0){
        // pixel is in the segment

        xm = (j > 1) * (L1dot(smooth(i,j-1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        xp = (j < p.cols - 2) * (L1dot(smooth(i,j+1), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 * alpha/(h*h);
        ym = (i > 1) * (L1dot(smooth(i-1,j), EPSILON_S) + L1dot(smooth(i,j), EPSILON_S))/2.0 *  alpha/(h*h);
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
             std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = getParameter("alpha", parameters);
   double kappa = getParameter("kappa", parameters);
   double omega = getParameter("omega", parameters);

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


void updatePhi(const cv::Mat_<double> &data_p,
               const cv::Mat_<double> &data_m,
               const cv::Mat_<double> &smooth_p,
               const cv::Mat_<double> &smooth_m,
               cv::Mat_<double> &phi,
               std::unordered_map<std::string, parameter> &parameters,
               const cv::Mat_<double> &mask,
               double h){

  // update the segment indicator function using implicit scheme

  double beta = getParameter("beta", parameters);
  double kappa = getParameter("kappa", parameters);
  double deltat = getParameter("deltat", parameters);

  double c1, c2, c3, c4, m, c, tmp;


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
                          -deltat*kappa*Hdot(kappa*phi(i,j))*(L1(data_p(i,j), EPSILON_D) - L1(data_m(i,j), EPSILON_D)));
    }
  }

}


double H(double x){
  //return (x >= 0) ? 1 : 0;
  return 0.5 * (1 + (2.0/M_PI)*std::atan(x/DELTA));
}

double Hdot(double x){
  return (1.0/M_PI) * (DELTA/(DELTA*DELTA + x*x));
}
