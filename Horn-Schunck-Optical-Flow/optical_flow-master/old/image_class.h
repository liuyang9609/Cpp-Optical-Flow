/**********************************************************
**** Class to save a image ********************************
**********************************************************/
#ifndef IMAGE_H
#define IMAGE_H


#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include "flow_utility.h"
#include "lodepng.h"

class BaseArray {
  // base class which defines resampling, resizing etc on one 2D-Field
  // it serves as a base for the flowfield and image classes
  // only has one data field and defines up/downsampling from this data field, as well as Getter and Setter for the field

  protected:
    std::vector< std::vector<double> > data;

    double Lanczos(double x, int a);
    double Gaussian(double x, double sigma);

  public:
    BaseArray();
    BaseArray(int sizex, int sizey);

    std::pair<int, int> Size();

    double operator() (int x, int y, bool mirror);
    std::vector<double>& operator[] (int x);
    double At (int x, int y, bool mirror);
    void Set (int x, int y, double value);
    void SetAll (double value);
    void SetTo (BaseArray *src);
    void SetTo (std::vector< std::vector<double> > src);

    std::vector< std::vector<double> > *getData();

    void Resize (int sizex, int sizey);
    void Resize (std::pair<int, int> size);
    void ResampleArea (int sizex, int sizey);
    void ResampleLanczos (int sizex, int sizey, int n);
    void GaussianSmooth (double sigma);

};


struct RGBA {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
};

class FlowField {

  public:
    BaseArray u, v;

    std::pair<int, int> Size();
    void Resize (int sizex, int sizey);
    void Resize (std::pair<int, int> size);
    void SetAll (double value);
    void ResampleArea (int sizex, int sizey);
    void ResampleLanczos (int sizex, int sizey, int n);
    std::vector< std::vector<RGBA> > getColorImage();
    void writeToPNG(std::string filename);
    void writeErrorToPNG(std::string filename, FlowField &t);
    double CalcAngularError(FlowField &t);
};



class Img: public BaseArray {

  private:
    std::vector< std::vector<double> > original;

  public:
    std::pair<int, int> SizeOriginal();
    std::vector< std::vector<double> > getOriginal();
    double AtOriginal(int x, int y, bool mirror);
    void SetOriginal (int x, int y, double value);
    void SetOriginalTo(std::vector< std::vector<double> > src);
    void ResizeOriginal(int sizex, int sizey);

    void GaussianSmoothOriginal(double sigma);
    void ResampleArea (int sizex, int sizey);
    void ResampleLanczos (int sizex, int sizey, int n);

    double Hx();
    double Hy();
};

#endif
