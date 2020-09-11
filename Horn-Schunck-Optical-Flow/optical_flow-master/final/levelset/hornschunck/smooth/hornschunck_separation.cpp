/**
  * Horn Schunck method with additional gradient constraint
*/

#include "../hornschunck_separation.hpp"

void computeFlowField(const cv::Mat &image1,
                      const cv::Mat &image2,
                      const GroundTruth &truth,
                      cv::Mat_<double> &segmentation,
                      std::unordered_map<std::string, parameter> &parameters,
                      bool interactive,
                      cv::FileStorage &scenario
                      ){

  // convert images into 64 bit floating point images
  cv::Mat i1, i2;
  i1 = image1.clone();
  i1.convertTo(i1, CV_64F);
  i2 = image2.clone();
  i2.convertTo(i2, CV_64F);

  double sigma = getParameter("sigma", parameters);
  double gamma = getParameter("gamma", parameters);
  int phi_iter = parameters.at("phi_iter").value;
  int iter_flow_before_phi = parameters.at("iter_flow_before_phi").value;
  int maxiter = parameters.at("maxiter").value;
  double h = 1;
  
  // Blurring
  cv::GaussianBlur(i1, i1, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2, i2, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // compute Brightness and Gradient Tensors
  cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, h) + gamma * ComputeGradientTensor(i1, i2, h);

  // create flowfield
  cv::Mat_<cv::Vec2d> flowfield(i1.size());
  cv::Mat_<cv::Vec2d> flowfield_p(i1.size());
  cv::Mat_<cv::Vec2d> flowfield_m(i1.size());

  flowfield = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(0,0);
  flowfield_m = cv::Vec2d(0,0);

  // create phi
  cv::Mat_<double> phi(i1.size());
  phi = 1;

  // initial segmentation
  cv::Mat_<cv::Vec2d> initialflow;
  scenario["initialflow"] >> initialflow;
  cv::Vec6d dominantmotion;

  if (segmentation.empty()) {
    segmentation.create(i1.size());
    segmentation = 0;
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


  cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

  // main loop
  for (int i = 0; i < maxiter; i++){
    for (int j = 0; j < iter_flow_before_phi; j++){
      HS_Stepfunction(t, flowfield_p, flowfield_m, phi, parameters, h);
    }
    for (int k = 0; k < phi_iter; k++){
      updatePhi(flowfield_p, flowfield_m, phi, t, parameters, h);
    }
  }

  // combine flowfield_p and flowfield_m depending on phi
  for (int i = 0; i < flowfield.rows; i++){
    for (int j = 0; j < flowfield.cols; j++){
      flowfield(i,j) = (phi(i,j) > 0) ? flowfield_p(i,j) : flowfield_m(i,j);
    }
  }
  flowfield = flowfield(cv::Rect(1,1,image1.cols, image1.rows));
  phi = phi(cv::Rect(1,1,image1.cols, image1.rows));

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
void HS_Stepfunction(const cv::Mat_<cv::Vec6d> &t,
                     cv::Mat_<cv::Vec2d> &flowfield_p,
                     cv::Mat_<cv::Vec2d> &flowfield_m,
                     cv::Mat_<double> &phi,
                     std::unordered_map<std::string, parameter> &parameters,
                     double h){

  updateU(flowfield_p, phi, t, parameters, h, 1.0);
  updateU(flowfield_m, phi, t, parameters, h, -1.0);


  updateV(flowfield_p, phi, t, parameters, h, 1.0);
  updateV(flowfield_m, phi, t, parameters, h, -1.0);

}


void updateU(cv::Mat_<cv::Vec2d> &flowfield,
             cv::Mat_<double> &phi,
             const cv::Mat_<cv::Vec6d> &t,
             std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = getParameter("alpha", parameters);
  double kappa = getParameter("kappa", parameters);
  double omega = getParameter("omega", parameters);


  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++){

      if (phi(i,j)*sign > 0){
        // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h) * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0;
        xm =  (j > 1) * alpha/(h*h) * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0;
        yp =  (i < flowfield.rows-2) * alpha/(h*h) * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0;
        ym =  (i > 1) * alpha/(h*h) * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0;
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          - H(kappa * phi(i,j) *sign) * (t(i, j)[4] + t(i, j)[3] * flowfield(i,j)[1])
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(H(kappa * phi(i,j)*sign) * t(i, j)[0] + sum);


      } else {
        // for now use smoothess term here

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h);
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(sum);

      }
    }
  }
}


void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               std::unordered_map<std::string, parameter> &parameters,
               double h,
               double sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = getParameter("alpha", parameters);
   double kappa = getParameter("kappa", parameters);
   double omega = getParameter("omega", parameters);


   for (int i = 1; i < flowfield.rows-1; i++){
     for (int j = 1; j < flowfield.cols-1; j++){

       if (phi(i,j)*sign > 0){
       // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h) * (H(phi(i,j+1)*sign) + H(phi(i,j)*sign))/2.0;
        xm =  (j > 1) * alpha/(h*h) * (H(phi(i,j-1)*sign) + H(phi(i,j)*sign))/2.0;
        yp =  (i < flowfield.rows-2) * alpha/(h*h) * (H(phi(i+1,j)*sign) + H(phi(i,j)*sign))/2.0;
        ym =  (i > 1) * alpha/(h*h) * (H(phi(i-1,j)*sign) + H(phi(i,j)*sign))/2.0;
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          - H(kappa * phi(i,j) *sign) * (t(i, j)[5] + t(i, j)[3] * flowfield(i,j)[0])
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(H(kappa * phi(i,j)*sign) * t(i, j)[1] + sum);

      } else {
        // pixel lies out of the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h);
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(sum);

      }
    }
  }
}




void updatePhi(cv::Mat_<cv::Vec2d> &flowfield_p,
               cv::Mat_<cv::Vec2d> &flowfield_m,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               std::unordered_map<std::string, parameter> &parameters,
               double h){

  // update the segment indicator function using implicit scheme

  double alpha = getParameter("alpha", parameters);
  double beta = getParameter("beta", parameters);
  double kappa = getParameter("kappa", parameters);
  double deltat = getParameter("deltat", parameters);

  double data, smooth, c1, c2, c3, c4, m, c, tmp;


  // compute derivatives (not very efficient, we could use one mat for each derivatives)
  cv::Mat_<double> f_p[2], f_m[2];
  cv::Mat_<double> ux_p, ux_m, uy_p, uy_m, vx_p, vx_m, vy_p, vy_m, phix, phiy, kernel;

  // split flowfields
  split(flowfield_p, f_p);
  split(flowfield_m, f_m);


  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(f_p[0], ux_p, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], ux_m, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vx_p, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vx_m, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phix, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);


  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(f_p[0], uy_p, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], uy_m, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vy_p, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vy_m, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phiy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  for (int i = 1; i < phi.rows-1; i++){
    for (int j = 1; j < phi.cols-1; j++){

      // compute the data term parts
      smooth = ux_p(i,j) * ux_p(i,j) + uy_p(i,j) * uy_p(i,j) + vx_p(i,j) * vx_p(i,j) + vy_p(i,j) * vy_p(i,j);
      smooth = smooth - ux_m(i,j) * ux_m(i,j) - uy_m(i,j) * uy_m(i,j) - vx_m(i,j) * vx_m(i,j) - vy_m(i,j) * vy_m(i,j);


      // compute smoothness term parts
      data =   t(i,j)[0] * f_p[0](i,j) * f_p[0](i,j)        // J11*du^2
             + t(i,j)[1] * f_p[1](i,j) * f_p[1](i,j)        // J22*dv^2
             + t(i,j)[2]                                    // J33
             + t(i,j)[3] * f_p[0](i,j) * f_p[1](i,j) * 2    // J21*du*dv
             + t(i,j)[4] * f_p[0](i,j) * 2                  // J13*du
             + t(i,j)[5] * f_p[1](i,j) * 2;                 // J23*dv

      data = data - t(i,j)[0] * f_m[0](i,j) * f_m[0](i,j)        // J11*du^2
                  - t(i,j)[1] * f_m[1](i,j) * f_m[1](i,j)        // J22*dv^2
                  - t(i,j)[2]                                    // J33
                  - t(i,j)[3] * f_m[0](i,j) * f_m[1](i,j) * 2    // J21*du*dv
                  - t(i,j)[4] * f_m[0](i,j) * 2                  // J13*du
                  - t(i,j)[5] * f_m[1](i,j) * 2;                 // J23*dv


      // terms with phi using semi-implicit scheme
      // using the vese chan discretization
      // also make sure that derivative on boundries is set to zero
      tmp = (j< phi.cols-2) * std::pow((phi(i,j+1) - phi(i,j))/h, 2) + (i>1)*(i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i-1,j))/(2*h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c1 = std::sqrt(1.0/(tmp + EPSILON_P));

      tmp = (j>1)*std::pow((phi(i,j) - phi(i,j-1))/h, 2) + (i<phi.rows-2)*(i>1)*(j>1)*std::pow((phi(i+1,j-1) - phi(i-1,j-1))/(2*h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c2 = std::sqrt(1.0/(tmp + EPSILON_P));

      tmp = (j>1)*(j<phi.cols-2)*std::pow((phi(i,j+1) - phi(i,j-1))/(2*h), 2) + (i<phi.rows-2)*std::pow((phi(i+1,j) - phi(i,j))/(h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c3 = std::sqrt(1.0/(tmp + EPSILON_P));

      tmp = (i>1)*(j>1)*(j<phi.cols-2)*std::pow((phi(i-1,j+1) - phi(i-1,j-1))/(2*h), 2) + (i>1)*std::pow((phi(i,j) - phi(i-1,j))/(h),2);
      tmp = (tmp < 0) ? 0 : tmp;
      c4 = std::sqrt(1.0/(tmp + EPSILON_P));

      m = (deltat*Hdot(phi(i,j))*beta)/(h*h);
      c = 1+m*(c1+c2+c3+c4);
      phi(i,j) = (1.0/c)*(phi(i,j) + m*(c1*phi(i,j+1)+c2*phi(i,j-1)+c3*phi(i+1,j)+c4*phi(i-1,j))
                          -deltat*kappa*Hdot(kappa*phi(i,j))*data
                          -deltat*alpha*Hdot(phi(i,j))*smooth);
    }
  }

}


double H(double x){
  return 0.5 * (1 + (2.0/M_PI)*std::atan(x/DELTA));
}

double Hdot(double x){
  return (1.0/M_PI) * (DELTA/(DELTA*DELTA + x*x));
}
