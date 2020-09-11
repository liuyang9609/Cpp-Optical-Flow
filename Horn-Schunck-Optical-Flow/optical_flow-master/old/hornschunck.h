
#ifndef HORNSCHUNCK_H
#define HORNSCHUNCK_H

#include "image_class.h"
#include <iostream>

void HornSchunckLevelLoop(int level, int iter, double alpha, double omega, double wrapfactor, Img &i1, Img &i2, FlowField &c, FlowField &d);
void HornSchunckOuterLoop(int iter,
                          double alpha,
                          double omega,
                          std::pair<int, int> size,
                          Img &i1,
                          Img &i2,
                          FlowField &c,
                          FlowField &d);
void HornSchunckInnerLoop(double alpha,
                          double omega,
                          Img &i1,
                          Img &i2,
                          BaseArray &J_11,
                          BaseArray &J_22,
                          BaseArray &J_33,
                          BaseArray &J_12,
                          BaseArray &J_13,
                          BaseArray &J_23,
                          FlowField &c,
                          FlowField &d);

void ComputeConstancyTensor (Img &i1,
                             Img &i2,
                             BaseArray &J_11,
                             BaseArray &J_22,
                             BaseArray &J_33,
                             BaseArray &J_12,
                             BaseArray &J_13,
                             BaseArray &J_23);

void WrapImage(Img &i1, Img &i2, FlowField &c);

#endif
