#ifndef JACOBI_H

#define JACOBI_H
#include <iostream>
#include <opencv2/opencv.hpp>

template <class T, int size=2, int type = CV_32F>
cv::Mat_<T> jacobi_iterate(cv::Mat_<T> A, cv::Mat_<T> b)
{
  if (A.rows != A.cols)
  {
    // Exception
  }
  cv::Mat_<T> A_diagonal = A.diag();
  cv::Mat_<T> diag_ones = cv::Mat::ones(A_diagonal.rows, 1, type) / A_diagonal;

  cv::Mat_<T> D_inv = cv::Mat::diag(diag_ones);

  cv::Mat_<T> R = cv::Mat_<T>(A);
  for (int i = 0; i < R.cols; ++i)
  {
    R.template at<T>(i,i) = 0;
  }

  cv::Mat_<T> H = -D_inv*R;
  cv::Mat_<T> G = D_inv*b;

  cv::Mat_<T> x;
  cv::Mat_<T> x1 = cv::Mat::zeros(b.rows, 1, type);

  for (int i = 0; i < 30; ++i)
  {
    x = x1;
    x1 = H * x + G;
  }
  
  std::cout << x1 << std::endl;

  return x1;
}




#endif /* end of include guard: JACOBI_H */
