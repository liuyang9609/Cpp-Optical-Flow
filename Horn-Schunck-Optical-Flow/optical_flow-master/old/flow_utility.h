#ifndef FLOW_UTILITY_H
#define FLOW_UTILITY_H

#include <string>
#include "image_class.h"

double BilinearInterpolation(double x1, double x2, double x3, double x4, double dx, double dy);
double AreaCovered(int cx, int cy, double xleft, double xright, double yleft, double yright);
double RGBtoGray(unsigned char r, unsigned char g, unsigned char b);
unsigned char byte_range(int number);

#endif
