/**
* this functions divides the initial flowfield into two segements,
* which are used as an initial segmentation for the algorithm
*/

#include "initial_segmentation.h"


void UltraSimpleInitialization(BaseArray &phi){
  int blocksize = 100;
  std::pair<int, int> size = phi.Size();

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      phi[i][j] = ((i / blocksize) % 2 > 0 ) ? 255.0 : -255.0;
    }
  }
}

void OriginalSegmentation(FlowField &f, BaseArray &phi, Img &image, int blockSize, double Tstage, double Tmerge, double Tassign){

  // number of blocks in each direction
  int numBlocksX = std::floor(image.Size().first/(float)blockSize);
  int numBlocksY = std::floor(image.Size().second/(float)blockSize);

  // array to decide if a block is reliable
  bool reliable[numBlocksX][numBlocksY];
  affine_parameters parameters[numBlocksX][numBlocksY];

  // loop over the blocks
  for (int i = 0; i < numBlocksX; i++){
    for (int j = 0; j < numBlocksY; j++){
      // compute affine parameters
      parameters[i][j] = getAffineParameters(image, f, i, j, blockSize);

      // check for reliability
      reliable[i][j] = isBlockReliable(f, i, j, parameters[i][j], blockSize, Tstage);
    }
  }

  // find dominant affine model

}

affine_parameters getAffineParameters(Img &image, FlowField &f, int i, int j, int blockSize){
  // computing the affine paramters for the block at position (i,j) with gradient descent for
  // least square problem

  double epsilon = 0.01;
  double derror = 1;
  double dp1, dp2, dp3;

  affine_parameters p = {1, 1, 1, 1, 1, 1};
  affine_parameters pOld = {0, 0, 0, 0, 0, 0};

  // loop until approximation is near minimal for u
  while (derror > epsilon){
    dp1 = 0;
    dp2 = 0;
    dp3 = 0;

    for (int x = i*blockSize; x < (i+1)*blockSize; x++){
      for (int y = j*blockSize; y < (j+1)*blockSize; y++){
        dp1 += 2 * (x * p.a1 + y * p.a2 + p.a3 - f.u(x, y, true)) + x;
        dp2 += 2 * (x * p.a1 + y * p.a2 + p.a3 - f.u(x, y, true)) + y;
        dp3 += 2 * (x * p.a1 + y * p.a2 + p.a3 - f.u(x, y, true)) + 1;
      }
    }

    // save old parameters
    pOld.a1 = p.a1;
    pOld.a2 = p.a2;
    pOld.a3 = p.a3;

    p.a1 -= dp1;
    p.a2 -= dp2;
    p.a3 -= dp3;

    derror = (std::abs(p.a1 - pOld.a1) + std::abs(p.a2 - pOld.a2) + std::abs(p.a3 - pOld.a3));
  }

  // loop until approximation is near minimal for v
  derror = 1;
  while (derror > epsilon){
    dp1 = 0;
    dp2 = 0;
    dp3 = 0;

    for (int x = i*blockSize; x < (i+1)*blockSize; x++){
      for (int y = j*blockSize; y < (j+1)*blockSize; y++){
        dp1 += 2 * (x * p.a4 + y * p.a5 + p.a6 - f.v(x, y, true)) + x;
        dp2 += 2 * (x * p.a4 + y * p.a5 + p.a6 - f.v(x, y, true)) + y;
        dp3 += 2 * (x * p.a4 + y * p.a5 + p.a6 - f.v(x, y, true)) + 1;
      }
    }

    // save old parameters
    pOld.a4 = p.a4;
    pOld.a5 = p.a5;
    pOld.a6 = p.a6;

    p.a4 -= dp1;
    p.a5 -= dp2;
    p.a6 -= dp3;

    derror = (std::abs(p.a4 - pOld.a4) + std::abs(p.a5 - pOld.a5) + std::abs(p.a6 - pOld.a6));
  }

  return p;
}

bool isBlockReliable(FlowField &f, int i, int j, affine_parameters &p, int blockSize, double Tstage){
  double error = 0;
  for (int x = 0; x < blockSize; x++){
    for (int y = 0; y < blockSize; y++){
      error += ( x * p.a1 + y * p.a2 + p.a3 - f.u(i*blockSize+x, j*blockSize+y, true));
      error += ( x * p.a4 + y * p.a5 + p.a6 - f.v(i*blockSize+x, j*blockSize+y, true));
    }
  }

  return error < Tstage;
}
