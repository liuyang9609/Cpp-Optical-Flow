/**
  * Horn Schunck method with additional gradient constraint
*/

#include "hornschunck_separation.hpp"
const double DELTA=1.0;


void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 55, 1000, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 15, 100, 10};
  parameter gamma = {"gamma", 990, 1000, 1000};
  parameter maxiter = {"maxiter", 40, 400, 1};
  parameter kappa = {"kappa", 100, 100, 100};
  parameter beta = {"beta", 20, 300, 10};
  parameter deltat = {"deltat", 100, 100, 100};
  parameter phi_iter = {"phi_iter", 15, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 1, 100, 1};


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
}


void computeFlowField(const cv::Mat &image1, const cv::Mat &image2, std::unordered_map<std::string, parameter> &parameters,
                         cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &phi){

  // convert images into 64 bit floating point images
  cv::Mat i1, i2;
  i1 = image1.clone();
  i1.convertTo(i1, CV_64F);
  i2 = image2.clone();
  i2.convertTo(i2, CV_64F);

  // Blurring
  double sigma = (double)parameters.at("sigma").value/parameters.at("sigma").divfactor;
  cv::GaussianBlur(i1, i1, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);
  cv::GaussianBlur(i2, i2, cv::Size(0,0), sigma, sigma, cv::BORDER_REFLECT);

  // compute Brightness and Gradient Tensors
  double gamma = (double)parameters.at("gamma").value/parameters.at("gamma").divfactor;
  cv::Mat_<cv::Vec6d> t = (1.0 - gamma) * ComputeBrightnessTensor(i1, i2, 1, 1) + gamma * ComputeGradientTensor(i1, i2, 1, 1);

  // create flowfield
  //flowfield.create(i1.size());
  cv::Mat_<cv::Vec2d> flowfield_p(i1.rows, i1.cols);
  cv::Mat_<cv::Vec2d> flowfield_m(i1.rows, i1.cols);

  //flowfield = cv::Vec2d(0,0);
  //flowfield_p = cv::Vec2d(0,0);
  //flowfield_m = cv::Vec2d(0,0);
  flowfield_p = cv::Vec2d(1,0);
  flowfield_m = flowfield.clone();


  cv::copyMakeBorder(flowfield, flowfield, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(flowfield_p, flowfield_p, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(flowfield_m, flowfield_m, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(phi, phi, 1, 1, 1, 1, cv::BORDER_REPLICATE, 0);
  cv::copyMakeBorder(t, t, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

  // main loop
  int phi_iter = parameters.at("phi_iter").value;
  int iter_flow_before_phi = parameters.at("iter_flow_before_phi").value;
  for (int i = 0; i < parameters.at("maxiter").value; i++){
    for (int j = 0; j < iter_flow_before_phi; j++){
      HS_Stepfunction(t, flowfield_p, flowfield_m, phi, parameters);
    }
    for (int k = 0; k < phi_iter; k++){
      updatePhi(flowfield_p, flowfield_m, phi, t, parameters, 1);
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
                     const std::unordered_map<std::string, parameter> &parameters){

  double h = 1;

  updateU(flowfield_p, phi, t, parameters, h, 1.0);
  updateU(flowfield_m, phi, t, parameters, h, -1.0);


  updateV(flowfield_p, phi, t, parameters, h, 1.0);
  updateV(flowfield_m, phi, t, parameters, h, -1.0);

}


void updateU(cv::Mat_<cv::Vec2d> &flowfield,
             cv::Mat_<double> &phi,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             double sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;


  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++){

      //if (phi(i,j)*sign > 0){
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


      /*} else {
        // for now use smoothess term here

        // test for borders
        xp =  (j < flowfield.cols-2) * 1.0/(h*h);
        xm =  (j > 1) * 1.0/(h*h);
        yp =  (i < flowfield.rows-2) * 1.0/(h*h);
        ym =  (i > 1) * 1.0/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(sum);

      }*/
    }
  }
}


void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h,
               double sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
   double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
   double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;


   for (int i = 1; i < flowfield.rows-1; i++){
     for (int j = 1; j < flowfield.cols-1; j++){

       //if (phi(i,j)*sign > 0){
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

      /*} else {
        // pixel lies out of the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * 1.0/(h*h);
        xm =  (j > 1) * 1.0/(h*h);
        yp =  (i < flowfield.rows-2) * 1.0/(h*h);
        ym =  (i > 1) * 1.0/(h*h);
        sum = xp + xm + yp + ym;

        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(sum);

      }*/
    }
  }
}

/*

void updateU(cv::Mat_<cv::Vec2d> &flowfield,
             cv::Mat_<double> &phi,
             const cv::Mat_<cv::Vec6d> &t,
             const std::unordered_map<std::string, parameter> &parameters,
             double h,
             int sign){

  // helper variables
  double xm, xp, ym, yp, sum;
  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;
  double H_s = 0;

  for (int i = 1; i < flowfield.rows-1; i++){
    for (int j = 1; j < flowfield.cols-1; j++){

      //if (phi(i,j)*sign > 0){
        // pixel is in the segment
        H_s = (phi(i,j) * sign > 0) ? 1 : 0;

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h) ;
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[0] = (1.0-omega) * flowfield(i,j)[0];
        flowfield(i,j)[0] += omega * (
          - H_s * (t(i, j)[4] + t(i, j)[3] * flowfield(i,j)[1])
          + yp * flowfield(i+1,j)[0]
          + ym * flowfield(i-1,j)[0]
          + xp * flowfield(i,j+1)[0]
          + xm * flowfield(i,j-1)[0]
        )/(H_s * t(i, j)[0] + sum);
    }
  }
}


void updateV(cv::Mat_<cv::Vec2d> &flowfield,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h,
               int sign){

   // helper variables
   double xm, xp, ym, yp, sum;
   double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
   double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
   double omega = (double)parameters.at("omega").value/parameters.at("omega").divfactor;
   double H_s = 0;
   for (int i = 1; i < flowfield.rows-1; i++){
     for (int j = 1; j < flowfield.cols-1; j++){

       H_s = (phi(i,j) * sign > 0) ? 1 : 0;
         // pixel is in the segment

        // test for borders
        xp =  (j < flowfield.cols-2) * alpha/(h*h);
        xm =  (j > 1) * alpha/(h*h);
        yp =  (i < flowfield.rows-2) * alpha/(h*h);
        ym =  (i > 1) * alpha/(h*h);
        sum = xp + xm + yp + ym;

        // u component
        flowfield(i,j)[1] = (1.0-omega) * flowfield(i,j)[1];
        flowfield(i,j)[1] += omega * (
          - H_s * (t(i, j)[5] + t(i, j)[3] * flowfield(i,j)[0])
          + yp * flowfield(i+1,j)[1]
          + ym * flowfield(i-1,j)[1]
          + xp * flowfield(i,j+1)[1]
          + xm * flowfield(i,j-1)[1]
        )/(H_s * t(i, j)[1] + sum);


    }
  }
}

*/

void updatePhi(cv::Mat_<cv::Vec2d> &flowfield_p,
               cv::Mat_<cv::Vec2d> &flowfield_m,
               cv::Mat_<double> &phi,
               const cv::Mat_<cv::Vec6d> &t,
               const std::unordered_map<std::string, parameter> &parameters,
               double h){

  // update the segment indicator function using implicit scheme

  double alpha = (double)parameters.at("alpha").value/parameters.at("alpha").divfactor;
  double beta = (double)parameters.at("beta").value/parameters.at("beta").divfactor;
  double kappa = (double)parameters.at("kappa").value/parameters.at("kappa").divfactor;
  double deltat = (double)parameters.at("deltat").value/parameters.at("deltat").divfactor;

  double data, smooth, phi_update, c1, c2, c3, c4, m, c, tmp;


  // compute derivatives (not very efficient, we could use one mat for each derivatives)
  cv::Mat_<double> f_p[2], f_m[2];
  cv::Mat_<double> ux_p, ux_m, uy_p, uy_m, vx_p, vx_m, vy_p, vy_m, phix, phiy, kernel;

  // split flowfields
  split(flowfield_p, f_p);
  split(flowfield_m, f_m);


  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(f_p[0], ux_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], ux_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vx_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vx_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phix, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);


  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(f_p[0], uy_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[0], uy_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_p[1], vy_p, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(f_m[1], vy_m, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(phi, phiy, CV_64F, kernel * 1.0/(2*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

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
