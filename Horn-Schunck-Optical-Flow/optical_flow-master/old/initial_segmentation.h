#ifndef INITIAL_SEGMENTATION_H
#define INITIAL_SEGMENTATION_H


#include "image_class.h"


struct affine_parameters {
  double a1, a2, a3, a4, a5, a6;
};

void OriginalSegmentation(FlowField &f, BaseArray &phi, Img &image, int blockSize, double Tstage, double Tmerge, double Tassign);
bool isBlockReliable(Img &image, int i, int j, int blockSize, double Tstage);
affine_parameters getAffineParameters(Img &image, int i, int j, int blockSize);


#endif
