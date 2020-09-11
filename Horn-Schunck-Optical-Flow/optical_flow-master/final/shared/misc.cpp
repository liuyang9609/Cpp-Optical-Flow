#include "misc.hpp"

/**
  * computes a color representation of the current flow
  * @param cv::Mat f Current Flowfield
  * @return cv::Mat A RGB image of the flowfield, visualized with hsv conversion
*/
void computeColorFlowFieldError(const cv::Mat_<cv::Vec2d> &flowfield, const GroundTruth &truth, cv::Mat &img){
  // helper Mats
  cv::Mat_<cv::Vec2d> f;
  cv::Mat flowfieldcomponents[2];
  cv::Mat magnitude, angle, hsv, _hsv[3];
  _hsv[2] = cv::Mat::ones(flowfield.size(), CV_64F) * 255;

  f = flowfield-truth.truthfield; 

  // compute color image from flowfield using hsv
  split(f, flowfieldcomponents);
  cv::cartToPolar(flowfieldcomponents[0], flowfieldcomponents[1], magnitude, angle, true);
  double max = 1;
  cv::minMaxIdx(magnitude, NULL, &max);
  if (max > 0){
    magnitude = magnitude * 1.0/max;
  }
  _hsv[0] = angle;
  //_hsv[1] = magnitude;
  _hsv[1].create(magnitude.size(), CV_64F);

  // only set magnitude if truthfield is valid
  for (int i = 0; i < _hsv[1].rows; i++) {
    for (int j = 0; j < _hsv[1].cols; j++) {
      _hsv[1].at<double>(i,j) = (truth.mask(i,j) == 1) ? magnitude.at<double>(i,j) : 0;
    }
  }

  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv, CV_32FC3);     // cannot convert directly to CV_8UC3, because max(angle) could be 360 degrees
  cvtColor(hsv, hsv, CV_HSV2BGR);
  hsv.convertTo(img, CV_8UC3);
}







void computeColorFlowField(const cv::Mat_<cv::Vec2d> &f, cv::Mat &img){
  // helper Mats
  cv::Mat flowfieldcomponents[2];
  cv::Mat magnitude, angle, hsv, _hsv[3];
  _hsv[1] = cv::Mat::ones(f.size(), CV_64F);

  // compute color image from flowfield using hsv
  split(f, flowfieldcomponents);
  cv::cartToPolar(flowfieldcomponents[0], flowfieldcomponents[1], magnitude, angle, true);
  double max = 1;
  cv::minMaxIdx(magnitude, NULL, &max);
  if (max > 0){
    magnitude = magnitude * 255.0/2;//max;
  }
  _hsv[0] = angle;
  _hsv[2] = magnitude;
  cv::merge(_hsv, 3, hsv);
  hsv.convertTo(hsv, CV_32FC3);     // cannot convert directly to CV_8UC3, because max(angle) could be 360 degrees
  cvtColor(hsv, hsv, CV_HSV2BGR);
  hsv.convertTo(img, CV_8UC3);
}




void TrackbarCallback(int value, void *userdata){
  parameter *p = static_cast<parameter*>(userdata);
  std::cout << p->name << ": " << std::floor((double)p->value*100*p->scale)/100 << std::endl;
}






void computeColorFlowField2(const cv::Mat_<cv::Vec2d> &flowfield, cv::Mat &img){

  //cv::Mat_<cv::Vec2d> flowfield = f;

  // make temporary array
  //cv::Mat img(flowfield.size(), CV_8UC3);

  double Pi = 3.141592653589793238463;;
  double amp;
  double phi;
  double alpha, beta;
  double x, y;

  for (int i = 0; i < flowfield.rows; i++){
    for (int j = 0; j < flowfield.cols; j++){
      x = flowfield(i, j)[0];
      y = flowfield(i, j)[1];

      /* determine amplitude and phase (cut amp at 1) */
      amp = sqrt (x * x + y * y);
      if (amp > 1) amp = 1;
      if (x == 0.0)
        if (y >= 0.0) phi = 0.5 * Pi;
        else phi = 1.5 * Pi;
      else if (x > 0.0)
        if (y >= 0.0) phi = atan (y/x);
        else phi = 2.0 * Pi + atan (y/x);
      else phi = Pi + atan (y/x);
      phi = phi / 2.0;

      // interpolation between red (0) and blue (0.25 * Pi)
      if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
        beta  = phi / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha *   0.0 + beta * 255.0));
      }
      if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
        beta  = (phi-0.125 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta *  64.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *   0.0 + beta *  64.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
      if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
        beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha *  64.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha *  64.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
        beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
      }
      // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
      if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
        beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 0.0   + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 0.0   + beta * 0.0));
      }
      // interpolation between yellow (0.75 * Pi) and red (Pi)
      if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
        beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img.at<cv::Vec3b>(i,j)[0] = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img.at<cv::Vec3b>(i,j)[1] = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
        img.at<cv::Vec3b>(i,j)[2] = (unsigned char)floor(amp * (alpha * 0.0   + beta *   0.0));
      }

      /* check RGB range */
      /*img.at<cv::Vec3b>(i,j)[0] = byte_range((int)img[i][j].r);
      img.at<cv::Vec3b>(i,j)[1] = byte_range((int)img[i][j].g);
      img.at<cv::Vec3b>(i,j)[2] = byte_range((int)img[i][j].b);*/
    }
  }

  cvtColor(img, img, CV_RGB2BGR);
  //return img;
}







void computeSegmentationImage(const cv::Mat_<double> &phi, cv::Mat &segmentation){
  double max, min;
  cv::minMaxIdx(phi, &min, &max);
  segmentation.create(phi.size(), CV_8U);
  for (int i = 0; i < phi.rows; i++){
    for (int j = 0; j < phi.cols; j++){
      segmentation.at<uchar>(i,j) = (phi(i,j) + std::abs(min)) * 255.0/(std::abs(max)+std::abs(min));
      //segmentation.at<uchar>(i,j) = (phi(i,j) > 0) ? 255 : 0;
      if (std::isnan(segmentation.at<uchar>(i,j)) || !std::isfinite(segmentation.at<uchar>(i,j))){
        std::cout << "error segementation" << std::endl;
      }
    }
  }
}



void computeSegmentationImageBW(const cv::Mat_<double> &phi, const cv::Mat_<uchar> &image1, cv::Mat &segmentation) {
  segmentation.create(phi.size(), CV_8U);
  for (int i = 0; i < phi.rows; i++) {
    for (int j = 0; j < phi.cols; j++) {
      segmentation.at<uchar>(i,j) = (phi(i,j) > 0) ? image1(i,j)* 0.2 + 0.8*255 : image1(i,j);
    }
  }
}


void loadParameters(cv::FileNode &node, std::unordered_map<std::string, parameter> &parameters) {
  cv::FileNodeIterator it = node.begin(), it_end = node.end();
  parameter temp;

  for ( ; it != it_end; ++it) {
    temp.name = (*it).name();
    temp.value = (int)(*it)["value"];
    temp.maxvalue = (int)(*it)["maxvalue"];
    temp.scale = (double)(*it)["scale"];
    parameters.insert(std::make_pair<std::string, parameter>(temp.name, temp));
  }
}

void saveParameters(cv::FileStorage &storage, std::unordered_map<std::string, parameter> &parameters){
  storage << "parameters" << "{";
  
  for (auto i: parameters) {
    storage << i.first << "{";
    storage << "value" << i.second.value << "maxvalue" << i.second.maxvalue << "scale" << i.second.scale;
    storage << "}";
  }

  storage << "}";
}

double getParameter(std::string name, std::unordered_map<std::string, parameter> &parameters){
  return parameters.at(name).value * parameters.at(name).scale;
}


void displayFlow(std::string windowname, const cv::Mat_<cv::Vec2d> &flowfield){
  cv::Mat flowfield_disp;
  computeColorFlowField(flowfield, flowfield_disp);
  cv::imshow(windowname, flowfield_disp);
}


void displayError(std::string windowname, const cv::Mat_<cv::Vec2d> &flowfield, const GroundTruth &truth){
  cv::Mat error_disp;
  computeColorFlowFieldError(flowfield, truth, error_disp);
  cv::imshow(windowname, error_disp);
}

void displayImage(std::string windowname, const cv::Mat &image){
  cv::Mat disp_img;

  image.convertTo(disp_img, CV_8UC3);
  cv::imshow(windowname, disp_img);
}

void displaySegmentation(std::string windowname, const cv::Mat_<double> &phi){
  cv::Mat disp_seg;

  computeSegmentationImage(phi, disp_seg);
  cv::imshow(windowname, disp_seg);
}


void displaySegmentationBW(std::string windowname, const cv::Mat_<double> &phi, const cv::Mat &image){
  cv::Mat disp_seg;

  computeSegmentationImageBW(phi, image, disp_seg);
  cv::imshow(windowname, disp_seg);
}
