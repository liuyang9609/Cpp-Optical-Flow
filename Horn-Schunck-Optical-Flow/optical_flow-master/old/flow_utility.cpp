#include <string>
#include <vector>
#include <iostream>
#include "image_class.h"
#include "lodepng.h"

double BilinearInterpolation(double x1, double x2, double x3, double x4, double dx, double dy){
  // perform bilinear interpolation on the edge point x1-x4
  // (with x1 is left top corner, x2 top right, x3 bottom left, x4 bottom right) and dx and dy as offsets
  // (with dx offset from x1 to the right and dy offset from x1 to the bottom)

  return (    dx) * (    dy) * x4 +
         (1 - dx) * (    dy) * x3 +
         (    dx) * (1 - dy) * x2 +
         (1 - dx) * (1 - dy) * x1;
}

double AreaCovered(int cx, int cy, double xleft, double xright, double ybottom, double ytop){
  double left = (xleft < cx) ? 0 : xleft-cx;
  double right = (xright > cx+1) ? 1 : xright-cx;
  double bottom = (ybottom < cy) ? 0 : ybottom-cy;
  double top = (ytop > cy+1) ? 1 : ytop-cy;
  return (right-left)*(top-bottom);
}

double RGBtoGray(unsigned char r, unsigned char g, unsigned char b){
  int ri = (int) r,
      gi = (int) g,
      bi = (int) b;
  return (ri + gi + bi)/(3.0);
}

unsigned char byte_range(int number){
  number = (number > 255) ? 255 : number;
  number = (number < 0) ? 0 : number;
  return (unsigned char)number;
}
