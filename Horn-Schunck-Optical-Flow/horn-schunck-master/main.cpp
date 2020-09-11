#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/opencv.hpp>
#include "jacobi.hpp"
#include <vector>
#include <cmath>

// ffmpeg -framerate 24 -i derivative/%03d.jpg out.mp4

using namespace cv;

template <class T>
struct neighbourhood;

template <class T, class U>
U dee_I_dee_x(const neighbourhood<T>& hood)
{
  // First 
  U ret = (hood.I_xd_y_t + hood.I_xd_y_td + hood.I_xd_yd_t + hood.I_xd_yd_td)/4
            - (hood.I_x_y_t + hood.I_x_y_td + hood.I_x_yd_t + hood.I_x_yd_td)/4;

  return ret;
}

template <class T, class U>
U dee_I_dee_y(const neighbourhood<T>& hood)
{
  U ret = (hood.I_x_yd_t + hood.I_x_yd_td + hood.I_xd_yd_t + hood.I_xd_yd_td)/4
            - (hood.I_x_y_t + hood.I_x_y_td + hood.I_xd_y_t + hood.I_xd_y_td)/4;

  return ret;
}

template <class T, class U>
U dee_I_dee_t(const neighbourhood<T>& hood)
{
  U ret = (hood.I_x_y_td + hood.I_x_yd_td + hood.I_xd_y_td + hood.I_xd_yd_td)/4
            - (hood.I_x_y_t + hood.I_x_yd_t + hood.I_xd_y_t + hood.I_xd_yd_t)/4;

  return ret;
}

template <class T>
struct neighbourhood
{
  T I_x_y_t;
  T I_xd_y_t;
  T I_xd_yd_t;
  T I_x_yd_t;

  T I_x_y_td;
  T I_xd_y_td;
  T I_xd_yd_td;
  T I_x_yd_td;
};

template <class T>
neighbourhood<T> fetch_hood(__uint16_t cols, MatConstIterator_<T> t0, MatConstIterator_<T> t1)
{
  T I_x_y_t = *t0;
  T I_xd_y_t = *(++t0);
  T I_xd_yd_t = *(t0+=cols);
  T I_x_yd_t = *(--t0);
  t0-=cols;

  T I_x_y_td = *t1;
  T I_xd_y_td = *(++t1);
  T I_xd_yd_td = *(t1+=cols);
  T I_x_yd_td = *(--t1);
  t1-=cols;

  neighbourhood<T> ret = {
    I_x_y_t,
    I_xd_y_t,
    I_xd_yd_t,
    I_x_yd_t,

    I_x_y_td,
    I_xd_y_td,
    I_xd_yd_td,
    I_x_yd_td
  };

  return ret;
}

float r(float var) 
{ 
    // 37.66666 * 100 =3766.66 
    // 3766.66 + .5 =37.6716    for rounding off value 
    // then type cast to int so value is 3766 
    // then divided by 100 so the value converted into 37.66 
    float value = (int)(var * 100 + .5); 
    return (float)value / 100; 
} 

int main()
{
  /**
   * Jacobi test
   * Uncomment
   *  std::vector<unsigned char> vec{2,1,2,3};
   *  std::vector<unsigned char> b{4,8};
   *  jacobi_iterate<float, CV_32F>(cv::Mat_<unsigned char>(2,2, vec.data()), cv::Mat_<unsigned char>(2, 1, b.data()));
   *  return 0;
  */

  std::cout << "Differentiating..." << std::endl;

  Mat image, image2;

  for (int i = 1; i < 542; ++i)
  {
    std::cout << "Differentiating " << i << " ..." << std::endl;

    std::stringstream path_im;
    if (image2.rows == 0)
    {
      path_im << "video/" << std::setfill('0') << std::setw(3) << i << ".jpg";
      image = imread( path_im.str(), IMREAD_COLOR );
    }
    else
    {
      image = image2;
    }

    path_im.str(std::string());
    path_im << "video/" << std::setfill('0') << std::setw(3) << (i + 1) << ".jpg";
    image2 = imread( path_im.str(), IMREAD_COLOR );

    Mat res(image);

    Mat grey, grey2;
    cvtColor(image, grey, COLOR_BGR2GRAY);
    cvtColor(image2, grey2, COLOR_BGR2GRAY);

    typedef unsigned char VT;
    typedef Vec<float, 3> DEE3T;
    typedef Vec<unsigned char, 3> V3T;
    typedef int B;

    MatConstIterator_<VT> ptr1 = grey.begin<VT>();
    MatConstIterator_<VT> end1 = grey.end<VT>();
    MatConstIterator_<VT> ptr2 = grey2.begin<VT>();
    MatConstIterator_<VT> end2 = grey2.end<VT>();

    Mat Dee(grey.rows, grey.cols, CV_32FC3, DEE3T(0,0,0));

    MatIterator_<DEE3T> dee_ptr = Dee.begin<DEE3T>();

    MatIterator_<V3T> ptr_res = res.begin<V3T>();

    Mat UV(grey.rows, grey.cols, CV_32FC3, DEE3T(0,0,0));
    Mat UV1(grey.rows, grey.cols, CV_32FC3, DEE3T(0,0,0));

    ++ptr1;
    ++ptr_res;
    ++dee_ptr;
    while (ptr1 != end1-1) 
    {
      neighbourhood<VT> hood = fetch_hood(res.cols, ptr1, ptr2);
      B I_x = dee_I_dee_x<VT, B>(hood);
      B I_y = dee_I_dee_y<VT, B>(hood);
      B I_t = dee_I_dee_t<VT, B>(hood);
      *ptr_res = V3T(abs(I_x), abs(I_y), abs(I_t));
      *dee_ptr = DEE3T(I_x, I_y, I_t);

      ++dee_ptr;
      ++ptr_res;
      ++ptr1;
      ++ptr2;
    }

    // Iterate
    for (int i = 0; i < 10; ++i)
    {
      // float u_max = 0;
      // float v_max = 0;
      UV = UV1;

      int UV_cols = UV.cols; 
      MatIterator_<DEE3T> UV_ptr = UV.begin<DEE3T>();
      MatIterator_<DEE3T> UV_end = UV.end<DEE3T>();
      MatIterator_<DEE3T> UV1_ptr = UV1.begin<DEE3T>();

      dee_ptr = Dee.begin<DEE3T>();

      dee_ptr+=(UV_cols+1);
      UV_ptr+=(UV_cols+1);
      UV1_ptr+=(UV_cols+1);

      float lambda = 0.01;

      std::cout << "Iter..." << std::endl;
      while (UV_ptr != UV_end-(UV_cols+1)) 
      {
        float u_bar = 0;
        float v_bar = 0;

        // Get the means
          // u-1
          DEE3T uv = *(--UV_ptr);
          u_bar += uv[0];
          v_bar += uv[1];

          // u+1
          uv = *(UV_ptr+=2);
          u_bar += uv[0];
          v_bar += uv[1];
          --UV_ptr;

          // v-1
          uv = *(UV_ptr-=UV_cols);
          u_bar += uv[0];
          v_bar += uv[1];

          // v+1
          uv = *(UV_ptr+=(2*UV_cols));
          u_bar += uv[0];
          v_bar += uv[1];
          UV_ptr-=UV_cols;

          u_bar /= 4;
          v_bar /= 4;
        //
         
        float I_x = (*dee_ptr)[0];
        float I_y = (*dee_ptr)[1];
        float I_t = (*dee_ptr)[2];

        float alpha = 0;

        float denom = (lambda*lambda + I_x*I_x + I_y*I_y);
        if (denom != 0)
        {
          alpha = (I_x * u_bar + I_y * v_bar + I_t) / denom;
        }

        float u_1 = r(u_bar - I_x * alpha);
        float v_1 = r(v_bar - I_y * alpha);
        // u_max = abs(u_1) > u_max ? abs(u_1) : u_max;
        // v_max = abs(v_1) > v_max ? abs(v_1) : v_max;

        //std::cout << u_1 << std::endl;

        *UV1_ptr = DEE3T(u_1, v_1, 0);

        // float magnitude = std::sqrt(u_1*u_1 + v_1*v_1);
        // 
        // float u_1_hat = magnitude != 0 ? u_1 / magnitude : 0;
        // float v_1_hat = magnitude != 0 ? v_1 / magnitude : 0;

        // *UV1_ptr = DEE3T(u_1_hat, v_1_hat, 0);

        ++UV_ptr;
        ++UV1_ptr;
        ++dee_ptr;
      }
      // std::cout << "u_max: " << u_max << std::endl;
      // std::cout << "v_max: " << v_max << std::endl;
    }

    // Uncomment to scale the output
    MatIterator_<DEE3T> UV1_ptr = UV1.begin<DEE3T>();
    MatIterator_<DEE3T> UV1_end = UV1.end<DEE3T>();
    while (UV1_ptr != UV1_end) 
    {
      DEE3T uv = *UV1_ptr;
      *UV1_ptr = DEE3T(abs(uv[0]), abs(uv[1]), 0);
      ++UV1_ptr;
    }

    path_im.str(std::string());
    path_im << "derivative/" << std::setfill('0') << std::setw(3) << i << ".jpg";

    // imwrite(path_im.str(), res);
    imwrite(path_im.str(), UV1);
  }

  std::cout << "Done..." << std::endl;
}
