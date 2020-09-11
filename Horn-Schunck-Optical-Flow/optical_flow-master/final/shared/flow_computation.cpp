
#include "flow_computation.hpp"

void remap_border(cv::Mat &image, const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat_<double> &mask, double h){
  // wrap the image with the information in flowfield. if information is out of boundaries, set mask to 0 (then the data term will not be taken into account)

  double x_float, y_float, x_int, y_int, x_delta, y_delta, value;
  
  // copy input image to temporary image
  cv::Mat image2 = image.clone();

  for (int i = 0; i < flowfield.rows; i++){
    for (int j = 0; j < flowfield.cols; j++){
      
      // compute new floating point location of image
      x_float = j + flowfield(i,j)[0] * 1.0/h;
      y_float = i + flowfield(i,j)[1] * 1.0/h;
      
      if ( (x_float < 0) || (x_float >= flowfield.cols-1) || (y_float < 0) || (y_float >= flowfield.rows-1) ){
        // flow is outsite of boundaries

        mask(i,j) = 0;
        value = image2.at<double>(i,j);

      } else {
        // flow is inside boundaries => wraping with bilinear interpolation
        mask(i,j) = 1.0;
        
        // compute integral and non-integral part of flow
        x_int = std::floor(x_float);
        y_int = std::floor(y_float);
        x_delta = x_float - x_int;
        y_delta = y_float - y_int;

        // perform bilinear interpolation
        value = (1.0-x_delta) * (1.0-y_delta) * image2.at<double>(y_int  , x_int  ) +
                (1.0-x_delta) *      y_delta  * image2.at<double>(y_int+1, x_int  ) +
                     x_delta  * (1.0-y_delta) * image2.at<double>(y_int  , x_int+1) +
                     x_delta  *      y_delta  * image2.at<double>(y_int+1, x_int+1);
      }

      image.at<double>(i, j) = value;
    }
  }
}



void computeSmoothnessTerm(const cv::Mat_<cv::Vec2d> &f, const cv::Mat_<cv::Vec2d> &p, cv::Mat_<double> &smooth, double h){
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
  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(flow_u, uy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(flow_v, vy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  // derivates in x-dirction
  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(flow_u, ux, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(flow_v, vx, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

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
  cv::copyMakeBorder(i1, i1, 1, 1, 1, 1, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED, 0);
  cv::copyMakeBorder(i2, i2, 1, 1, 1, 1, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED, 0);

  // remap i2
  remap_border(i2, f, mask, h);

  // compute fx and fy for image 1 and image2
  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(i1, i1x, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(i2, i2x, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(i1, i1y, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(i2, i2y, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  for (int i = 0; i < f.rows; i++) {
    for (int j = 0; j < f.cols; j++) {
      data(i,j) = (1-gamma) * std::pow(i2.at<double>(i,j) - i1.at<double>(i,j),2);
      data(i,j) += gamma * std::pow(i2x.at<double>(i,j) - i1x.at<double>(i,j), 2);
      data(i,j) += gamma * std::pow(i2y.at<double>(i,j) - i1y.at<double>(i,j), 2);
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
