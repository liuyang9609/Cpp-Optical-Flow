#include "tensor_computation.hpp"
/*
cv::Mat ComputeGradientTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double hx, double hy){

  // for now give hx and hy as parameters, maybe later calculate them with ROI
  cv::Mat middle, kernel, t, x, y, xx, yy, xy, xt, yt;

  middle = 0.5 * i1 + 0.5 * i2;
  t = i2 - i1;

  kernel = (cv::Mat_<double>(1,3) << 1, -2, 1);
  cv::filter2D(middle, xx, -1, kernel * 1.0/(hx*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(t, xt, CV_64F, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  kernel = (cv::Mat_<double>(3,1) << 1, -2, 1);
  cv::filter2D(middle, yy, CV_64F, kernel * 1.0/(hy*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(t, yt, CV_64F, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  kernel = (cv::Mat_<double>(3,3) << 1, 0, -1, 0, 0, 0, -1, 0, 1);
  cv::filter2D(middle, xy, CV_64F, kernel * 1.0/(4*hx*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  // channel 0=J11, 1=J22, 2=J33, 3=J12, 4=J13, 5=J23
  cv::Mat_<cv::Vec6d> b(i1.size());
  for (int i = 0; i < b.rows; i++){
    for (int j = 0; j < b.cols; j++){
      b(i,j)[0] = xx.at<double>(i,j) * xx.at<double>(i,j) + xy.at<double>(i,j) * xy.at<double>(i,j);
      b(i,j)[1] = xy.at<double>(i,j) * xy.at<double>(i,j) + yy.at<double>(i,j) * yy.at<double>(i,j);
      b(i,j)[2] = xt.at<double>(i,j) * xt.at<double>(i,j) + yt.at<double>(i,j) * yt.at<double>(i,j);
      b(i,j)[3] = xx.at<double>(i,j) * xy.at<double>(i,j) + xy.at<double>(i,j) * yy.at<double>(i,j);
      b(i,j)[4] = xx.at<double>(i,j) * xt.at<double>(i,j) + xy.at<double>(i,j) * yt.at<double>(i,j);
      b(i,j)[5] = xy.at<double>(i,j) * xt.at<double>(i,j) + yy.at<double>(i,j) * yt.at<double>(i,j);
    }
  }

  return b;
}

cv::Mat ComputeBrightnessTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double hx, double hy){

  // compute derivatives
  cv::Mat x, y, t, kernel, middle;
  t = i2 - i1;
  middle = 0.5 * i1 + 0.5 * i2;

  kernel = (cv::Mat_<double>(1,3) << -1, 0, 1);
  cv::filter2D(middle, x, -1, kernel * 1.0/(2*hx), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  kernel = (cv::Mat_<double>(3,1) << -1, 0, 1);
  cv::filter2D(middle, y, -1, kernel * 1.0/(2*hy), cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

  // compute tensor
  // channel 0=J11, 1=J22, 2=J33, 3=J12, 4=J13, 5=J23
  cv::Mat_<cv::Vec6d> b(i1.size());
  for (int i = 0; i < i1.rows; i++){
    for (int j = 0; j < i1.cols; j++){
      b(i,j)[0] = x.at<double>(i,j) * x.at<double>(i,j);
      b(i,j)[1] = y.at<double>(i,j) * y.at<double>(i,j);
      b(i,j)[2] = t.at<double>(i,j) * t.at<double>(i,j);
      b(i,j)[3] = x.at<double>(i,j) * y.at<double>(i,j);
      b(i,j)[4] = x.at<double>(i,j) * t.at<double>(i,j);
      b(i,j)[5] = y.at<double>(i,j) * t.at<double>(i,j);

    }
  }

  return b;
}
*/


cv::Mat ComputeGradientTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double h){
  // for now give hx and hy as parameters, maybe later calculate them with ROI
  cv::Mat middle, kernel, t, x, y, xx, yy, xy, yx, xt, yt;

  middle = 0.5 * i1 + 0.5 * i2;
  t = i2 - i1;

  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(middle, x, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(x, xx, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(t, xt, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(middle, y, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(y, yy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(x, xy, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  cv::filter2D(t, yt, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
  
  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(y, yx, CV_64F, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  xy = 0.5 * xy + 0.5 * yx;

  // channel 0=J11, 1=J22, 2=J33, 3=J12, 4=J13, 5=J23
  cv::Mat_<cv::Vec6d> b(i1.size());
  for (int i = 0; i < b.rows; i++){
    for (int j = 0; j < b.cols; j++){
      b(i,j)[0] = xx.at<double>(i,j) * xx.at<double>(i,j) + xy.at<double>(i,j) * xy.at<double>(i,j);
      b(i,j)[1] = xy.at<double>(i,j) * xy.at<double>(i,j) + yy.at<double>(i,j) * yy.at<double>(i,j);
      b(i,j)[2] = xt.at<double>(i,j) * xt.at<double>(i,j) + yt.at<double>(i,j) * yt.at<double>(i,j);
      b(i,j)[3] = xx.at<double>(i,j) * xy.at<double>(i,j) + xy.at<double>(i,j) * yy.at<double>(i,j);
      b(i,j)[4] = xx.at<double>(i,j) * xt.at<double>(i,j) + xy.at<double>(i,j) * yt.at<double>(i,j);
      b(i,j)[5] = xy.at<double>(i,j) * xt.at<double>(i,j) + yy.at<double>(i,j) * yt.at<double>(i,j);
    }
  }

  return b;
}

cv::Mat ComputeBrightnessTensor(const cv::Mat_<double> &i1, const cv::Mat_<double> &i2, double h){

  // compute derivatives
  cv::Mat x, y, t, kernel, middle;
  t = i2 - i1;
  middle = 0.5 * i1 + 0.5 * i2;

  kernel = (cv::Mat_<double>(1,5) << 1, -8, 0, 8, -1);
  cv::filter2D(middle, x, -1, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  kernel = (cv::Mat_<double>(5,1) << 1, -8, 0, 8, -1);
  cv::filter2D(middle, y, -1, kernel * 1.0/(12*h), cv::Point(-1,-1), 0, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);

  // compute tensor
  // channel 0=J11, 1=J22, 2=J33, 3=J12, 4=J13, 5=J23
  cv::Mat_<cv::Vec6d> b(i1.size());
  for (int i = 0; i < i1.rows; i++){
    for (int j = 0; j < i1.cols; j++){
      b(i,j)[0] = x.at<double>(i,j) * x.at<double>(i,j);
      b(i,j)[1] = y.at<double>(i,j) * y.at<double>(i,j);
      b(i,j)[2] = t.at<double>(i,j) * t.at<double>(i,j);
      b(i,j)[3] = x.at<double>(i,j) * y.at<double>(i,j);
      b(i,j)[4] = x.at<double>(i,j) * t.at<double>(i,j);
      b(i,j)[5] = y.at<double>(i,j) * t.at<double>(i,j);

    }
  }

  return b;
}
