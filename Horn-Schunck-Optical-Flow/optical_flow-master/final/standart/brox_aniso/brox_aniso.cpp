/**
  * Method as defined by Brox et al.

*/

#include "brox_aniso.hpp"


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
    cv::Mat_<cv::Vec3d> smooth(partial.size());
    int nonlinear_step = parameters.at("nonlinear_step").value;
    
    for (int i = 0; i < maxiter; i++){
      if (i % nonlinear_step == 0 || i == 0){
        computeDataTerm(partial, t, data);
        computeAnisotropicSmoothnessTerm(flowfield, partial, smooth, h);
      }
      Brox_step_aniso_smooth(t, flowfield, partial, data, smooth, mask, parameters, h);
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
      std::cout << std::endl << "AAE:" << truth.computeAngularError(flowfield) << std::endl;
      std::cout << "EPE:" << truth.computeEndpointError(flowfield) << std::endl;
    }
  }

}



void Brox_step_aniso_smooth(const cv::Mat_<cv::Vec6d> &t,
               const cv::Mat_<cv::Vec2d> &f,
               cv::Mat_<cv::Vec2d> &p,
               cv::Mat_<double> &data,
               cv::Mat_<cv::Vec3d> &smooth,
               cv::Mat_<double> &mask,
               std::unordered_map<std::string, parameter> &parameters,
               double h){

  // get parameters
  double alpha = getParameter("alpha", parameters);
  double omega = getParameter("omega", parameters);

  // helper variables
  double xm, xp, ym, yp, sum, tmp;

  // update partial flow field
  for (int i = 1; i < p.rows - 1; i++){
    for (int j = 1; j < p.cols - 1; j++){

      // handle borders for terms on von Neumann neighboorhood
      // isotropic part + anisotropic part

      xm = (j > 1) * (smooth(i,j-1)[0] + smooth(i,j)[0])/2.0 * alpha/(h*h);
      xp = (j < p.cols - 2) * (smooth(i,j+1)[0] + smooth(i,j)[0])/2.0 * alpha/(h*h);
      ym = (i > 1) * (smooth(i-1,j)[2] + smooth(i,j)[2])/2.0 * alpha/(h*h);
      yp = (i < p.rows - 2) * (smooth(i+1,j)[2] + smooth(i,j)[2])/2.0 * alpha/(h*h);
      sum = xm + xp + ym + yp;

      // compute du
      // data terms
      tmp = mask(i,j) * L1dot(data(i,j), EPSILON_D) * (t(i,j)[3] * p(i,j)[1] + t(i,j)[4]);

      // smoothness terms (von Neumann neighboorhood)
      tmp = tmp
            - xm * (f(i,j-1)[0] + p(i,j-1)[0])
            - xp * (f(i,j+1)[0] + p(i,j+1)[0])
            - ym * (f(i-1,j)[0] + p(i-1,j)[0])
            - yp * (f(i+1,j)[0] + p(i+1,j)[0])
            + sum * f(i,j)[0];

      // remaining neighboorhoods
      tmp += (j < p.cols - 2) * ( i > 1) * (i < p.rows - 2) *
            -(alpha/(h*h)) * (smooth(i,j+1)[1]+smooth(i,j)[1])/8.0 *
            (f(i+1,j+1)[0] + p(i+1,j+1)[0] + f(i+1,j)[0] + p(i+1,j)[0] - f(i-1,j)[0] - p(i-1,j)[0] - f(i-1,j+1)[0] - p(i-1,j+1)[0]);
      tmp -= (j > 1) * (i > 1) * ( i < p.rows - 2) *
            -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i,j-1)[1])/8.0 *
            (f(i+1,j)[0] + p(i+1,j)[0] + f(i+1,j-1)[0] + p(i+1,j-1)[0] - f(i-1,j)[0] - p(i-1,j)[0] - f(i-1,j-1)[0] - p(i-1,j-1)[0]);
      tmp += (i < p.rows - 2) * (j > 1) * (j < p.cols - 2) *
            -(alpha/(h*h)) * (smooth(i+1,j)[1] + smooth(i,j)[1])/8.0 *
            (f(i+1,j+1)[0] + p(i+1,j+1)[0] + f(i,j+1)[0] + p(i,j+1)[0] - f(i,j-1)[0] - p(i,j-1)[0] - f(i+1,j-1)[0] - p(i+1,j-1)[0]);
      tmp -= (i > 1) * (j > 1) * (j < p.cols - 2) *
            -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i-1,j)[1])/8.0 *
            (f(i,j+1)[0] + p(i,j+1)[0] + f(i-1,j+1)[0] + p(i-1,j+1)[0] - f(i,j-1)[0] - p(i,j-1)[0] - f(i-1,j-1)[0] - p(i-1,j-1)[0]);

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

      // remaining neighboorhoods
      tmp += (j < p.cols - 2) * ( i > 1) * (i < p.rows - 2) *
            -(alpha/(h*h)) * (smooth(i,j+1)[1]+smooth(i,j)[1])/8.0 *
            (f(i+1,j+1)[1] + p(i+1,j+1)[1] + f(i+1,j)[1] + p(i+1,j)[1] - f(i-1,j)[1] - p(i-1,j)[1] - f(i-1,j+1)[1] - p(i-1,j+1)[1]);
      tmp -= (j > 1) * (i > 1) * ( i < p.rows - 2) *
            -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i,j-1)[1])/8.0 *
            (f(i+1,j)[1] + p(i+1,j)[1] + f(i+1,j-1)[1] + p(i+1,j-1)[1] - f(i-1,j)[1] - p(i-1,j)[1] - f(i-1,j-1)[1] - p(i-1,j-1)[1]);
      tmp += (i < p.rows - 2) * (j > 1) * (j < p.cols - 2) *
            -(alpha/(h*h)) * (smooth(i+1,j)[1] + smooth(i,j)[1])/8.0 *
            (f(i+1,j+1)[1] + p(i+1,j+1)[1] + f(i,j+1)[1] + p(i,j+1)[1] - f(i,j-1)[1] - p(i,j-1)[1] - f(i+1,j-1)[1] - p(i+1,j-1)[1]);
      tmp -= (i > 1) * (j > 1) * (j < p.cols - 2) *
            -(alpha/(h*h)) * (smooth(i,j)[1] + smooth(i-1,j)[1])/8.0 *
            (f(i,j+1)[1] + p(i,j+1)[1] + f(i-1,j+1)[1] + p(i-1,j+1)[1] - f(i,j-1)[1] - p(i,j-1)[1] - f(i-1,j-1)[1] - p(i-1,j-1)[1]);

      // normalization
      tmp = tmp /(- mask(i,j) * L1dot(data(i,j), EPSILON_D) * t(i,j)[1] - sum);
      p(i,j)[1] = (1.0-omega) * p(i,j)[1] + omega * tmp;

    }
  }
}


void computeAnisotropicSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<cv::Vec3d> &smooth, double h){
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
  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(flow_u, uy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  // derivates in x-dirction
  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(flow_u, ux, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(flow_v, vx, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

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
      eigenvalues.at<double>(0) = L1dot(eigenvalues.at<double>(0), EPSILON_S);
      eigenvalues.at<double>(1) = L1dot(eigenvalues.at<double>(1), EPSILON_S);

      // recompute array with scaled eigenvalues
      tmparray = eigenvectors.inv() * cv::Mat::diag(eigenvalues) * eigenvectors;
      smooth(i,j)[0] = tmparray.at<double>(0,0);
      smooth(i,j)[1] = tmparray.at<double>(0,1);
      smooth(i,j)[2] = tmparray.at<double>(1,1);
    }
  }
}


/* ------------------------------------------------------------------------- */
/*void set_up_differential_operator_all_2d(
  fptype **s11,           
  fptype **s12,          
  fptype **s22,         
  fptype ***psi_pri_s_nb,
  itype  nx,            
  itype  ny,           
  itype  bx,          
  itype  by,         
  fptype hx,        
  fptype hy,       
  fptype m_alpha, 
  itype  n_theta 
)

// sets up differential operator (in smoothness term)
          
{
itype    i,j;                 // loop variables 
fptype   rxx,rxy,ryy;         // time saver 
fptype   **theta;             // discretisation coefficient 

  
// compute time saver variables 
 rxx  = m_alpha / (2.0 * hx * hx);
 ryy  = m_alpha / (2.0 * hy * hy);
 rxy  = m_alpha / (4.0 * hx * hy);
 
 
 // chose discretiation 
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &theta);
 
 if(n_theta==-1)
     for (i=bx; i<nx+bx; i++)    
     for (j=by; j<ny+by; j++)     
         theta[i][j]= -1.0;
 
 if(n_theta==0)
     for (i=bx; i<nx+bx; i++)    
     for (j=by; j<ny+by; j++)      
         theta[i][j]= 0.0;
 
 if(n_theta==1)
     for (i=bx; i<nx+bx; i++)    
     for (j=by; j<ny+by; j++)
         theta[i][j]=1.0;
 
 if(n_theta==2)
     for (i=bx; i<nx+bx; i++)    
     for (j=by; j<ny+by; j++)     
         theta[i][j]=signum(s12[i][j]);
 
 
 // struture of psi_pri_s_nb : 
 // psi_pri_s_nb[j][0]  : neighbour -> xmym  
 // psi_pri_s_nb[j][1]  : neighbour -> xm   
 // psi_pri_s_nb[j][2]  : neighbour -> xmyp
 // psi_pri_s_nb[j][3]  : neighbour ->   ym 
 // psi_pri_s_nb[j][4]  : neighbour ->   yp
 // psi_pri_s_nb[j][5]  : neighbour -> xpym 
 // psi_pri_s_nb[j][6]  : neighbour -> xp  
 // psi_pri_s_nb[j][7]  : neighbour -> xpyp 
 // psi_pri_s_nb[j][8]  : central pixel  
 
 
 for (i=bx; i<=bx; i++)    
     for (j=by; j<=by; j++)
     {           
     psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j+1])*rxy*s12[i][j+1];
     
     psi_pri_s_nb[i][j][5] = 0.0;
     
     psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j]) 
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][7] =   
         + (1+theta[i][j+1])*rxy*s12[i][j+1]
         + (1+theta[i+1][j])*rxy*s12[i+1][j];
     
     }
 
 for (i=bx; i<=bx; i++)    
     for (j=by+1; j<ny+by-1; j++)
     {           
         psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j+1])*rxy*s12[i][j+1];
         
         psi_pri_s_nb[i][j][5] = 
         - (1-theta[i][j-1])*rxy*s12[i][j-1]
         - (1-theta[i+1][j])*rxy*s12[i+1][j];
         
         psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j]) 
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i+1][j])*rxy*s12[i+1][j]
         - (1+theta[i+1][j])*rxy*s12[i+1][j];
         
         psi_pri_s_nb[i][j][7] =   
         + (1+theta[i][j+1])*rxy*s12[i][j+1]
         + (1+theta[i+1][j])*rxy*s12[i+1][j];
         
     }
 
 for (i=bx; i<=bx; i++)    
     for (j=ny+by-1; j<=ny+by-1; j++)
     {           
     psi_pri_s_nb[i][j][4] = 0.0;
     
     psi_pri_s_nb[i][j][5] = 
         - (1-theta[i][j-1])*rxy*s12[i][j-1]
         - (1-theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j])          
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][7] = 0.0;                 
     }
 
 for (i=bx+1; i<nx+bx-1; i++)    
     for (j=by; j<=by; j++)
     {           
     psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i][j+1])*rxy*s12[i][j+1]
         - (1+theta[i][j+1])*rxy*s12[i][j+1];
     
     psi_pri_s_nb[i][j][5] = 0.0; 
     
     psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j]) 
         + (1-theta[i][j])  *rxy*s12[i][j]         
         - (1+theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][7] =   
         + (1+theta[i][j+1])*rxy*s12[i][j+1]
         + (1+theta[i+1][j])*rxy*s12[i+1][j];
         
     }
 
 for (i=bx+1; i<nx+bx-1; i++)    
     for (j=by+1; j<ny+by-1; j++)
     {           
     psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i][j+1])*rxy*s12[i][j+1]
         - (1+theta[i][j+1])*rxy*s12[i][j+1];
     
     psi_pri_s_nb[i][j][5] = 
         - (1-theta[i][j-1])*rxy*s12[i][j-1]
         - (1-theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j]) 
         + (1-theta[i][j])  *rxy*s12[i][j]
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i+1][j])*rxy*s12[i+1][j]
         - (1+theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][7] =    
         + (1+theta[i][j+1])*rxy*s12[i][j+1]
         + (1+theta[i+1][j])*rxy*s12[i+1][j];
     
     }
 
 for (i=bx+1; i<nx+bx-1; i++)    
     for (j=by+ny-1; j<=ny+by-1; j++)
     {           
     psi_pri_s_nb[i][j][4] = 0.0;
     
     psi_pri_s_nb[i][j][5] = 
         - (1-theta[i][j-1])*rxy*s12[i][j-1]
         - (1-theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][6] = rxx * (s11[i+1][j] + s11[i][j])          
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i+1][j])*rxy*s12[i+1][j];
     
     psi_pri_s_nb[i][j][7] = 0.0;              
     }
 
 for (i=bx+nx-1; i<=nx+bx-1; i++)    
     for (j=by; j<=by; j++)
     {           
     psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])         
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i][j+1])*rxy*s12[i][j+1];
     
     psi_pri_s_nb[i][j][5] = 0.0;
     
     psi_pri_s_nb[i][j][6] = 0.0;
     
     psi_pri_s_nb[i][j][7] = 0.0;              
     }
 
 for (i=bx+nx-1; i<=nx+bx-1; i++)    
     for (j=by+1; j<ny+by-1; j++)
     {           
     psi_pri_s_nb[i][j][4] = ryy * (s22[i][j+1] + s22[i][j])         
         - (1+theta[i][j])  *rxy*s12[i][j]
         + (1-theta[i][j+1])*rxy*s12[i][j+1];
     
     psi_pri_s_nb[i][j][5] = 0.0;
     
     psi_pri_s_nb[i][j][6] = 0.0;
     
     psi_pri_s_nb[i][j][7] = 0.0;               
     }
 
 for (i=bx+nx-1; i<=nx+bx-1; i++)    
     for (j=by+ny-1; j<=ny+by-1; j++)
     {           
     psi_pri_s_nb[i][j][4] = 0.0;
     
     psi_pri_s_nb[i][j][5] = 0.0;
     
     psi_pri_s_nb[i][j][6] = 0.0; 
     
     psi_pri_s_nb[i][j][7] = 0.0;         
     }
 
 
 FREE_MATRIX(1, nx+2*bx, ny+2*by, theta);
 
 // mirror boundaries in y direction 
 for (i=0; i<nx+2*bx-1; i++)   
     for (j=1; j<=by; j++)  
     {
     psi_pri_s_nb[i][by-j][4]=0.0;
     psi_pri_s_nb[i][by-j][5]=0.0;
     psi_pri_s_nb[i][by-j][6]=0.0;
     psi_pri_s_nb[i][by-j][7]=0.0;
     psi_pri_s_nb[i][ny+by-1+j][4]=0.0;
     psi_pri_s_nb[i][ny+by-1+j][5]=0.0;
     psi_pri_s_nb[i][ny+by-1+j][6]=0.0;
     psi_pri_s_nb[i][ny+by-1+j][7]=0.0;
     }
 // mirror boundaries in x direction
 for (i=1; i<=bx; i++)   
     for (j=0; j<ny+2*by; j++)  
     {
     psi_pri_s_nb[bx-i][j][4]=0.0;
     psi_pri_s_nb[bx-i][j][5]=0.0;
     psi_pri_s_nb[bx-i][j][6]=0.0;
     psi_pri_s_nb[bx-i][j][7]=0.0;
     psi_pri_s_nb[nx+bx-1+i][j][4]=0.0;
     psi_pri_s_nb[nx+bx-1+i][j][5]=0.0;
     psi_pri_s_nb[nx+bx-1+i][j][6]=0.0;
     psi_pri_s_nb[nx+bx-1+i][j][7]=0.0;
     
     }
 
 // set up lower diagonal entries using symmetry property of A 
 // (use upper diagonal diffusion entries) 
 for (i=bx; i<nx+bx; i++)
     for (j=by; j<ny+by; j++)    
     {
     psi_pri_s_nb[i][j][0] =  psi_pri_s_nb[i-1][j-1][7];
     psi_pri_s_nb[i][j][1] =  psi_pri_s_nb[i-1][j  ][6];
     psi_pri_s_nb[i][j][2] =  psi_pri_s_nb[i-1][j+1][5];
     psi_pri_s_nb[i][j][3] =  psi_pri_s_nb[i  ][j-1][4];
     }
 
 // compute central enties (diffusion) 
 for (i=bx; i<nx+bx; i++)
     for (j=by; j<ny+by; j++)
     {       
     psi_pri_s_nb[i][j][8]= -( psi_pri_s_nb[i][j][0]+psi_pri_s_nb[i][j][1]
                  +psi_pri_s_nb[i][j][2]+psi_pri_s_nb[i][j][3]
                  +psi_pri_s_nb[i][j][4]+psi_pri_s_nb[i][j][5]
                  +psi_pri_s_nb[i][j][6]+psi_pri_s_nb[i][j][7]);
     } 
 */
 /*
 for (i=bx; i<nx+bx; i++)
   for (j=by; j<ny+by; j++)        
     {  
       if((i==bx+nx/2)&&(j==by+ny/2))
     {
       printf("\n WITH WARPING");   
       printf("\n Alpha   :  %f", m_alpha);
       printf("\n Theta   :  %d", n_theta);
       printf("\n hx      :  %f", hx);
       printf("\n hy      :  %f", hy);
       printf("\n Value 11:  %f", s11[i][j]);
       printf("\n Value 12:  %f", s12[i][j]);
       printf("\n Value 13:  %f", s22[i][j]);
       printf("\n Value  0:  %f", psi_pri_s_nb[i][j][0]);
       printf("\n Value  1:  %f", psi_pri_s_nb[i][j][1]);
       printf("\n Value  2:  %f", psi_pri_s_nb[i][j][2]);
       printf("\n Value  3:  %f", psi_pri_s_nb[i][j][3]);
       printf("\n Value  4:  %f", psi_pri_s_nb[i][j][4]);
       printf("\n Value  5:  %f", psi_pri_s_nb[i][j][5]);
       printf("\n Value  6:  %f", psi_pri_s_nb[i][j][6]);
       printf("\n Value  7:  %f", psi_pri_s_nb[i][j][7]);
     }
     } 
 */

// return;
//} 
