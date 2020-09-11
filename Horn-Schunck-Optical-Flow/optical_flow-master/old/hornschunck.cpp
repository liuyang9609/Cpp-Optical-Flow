#include "hornschunck.h"
#include "filehandling.h"

void ComputeConstancyTensor(Img &i1,
                            Img &i2,
                            BaseArray &J_11,
                            BaseArray &J_22,
                            BaseArray &J_33,
                            BaseArray &J_12,
                            BaseArray &J_13,
                            BaseArray &J_23){

  // make sure images have same dimensions
  std::pair<int, int> size = i1.Size();
  if (size != i2.Size()){
    std::cout << "image dimensions dont match in ComputeConstancyTensor" << std::endl;
    std::exit(1);
  }

  // get stepsize
  double hx = i1.Hx();
  double hy = i2.Hy();

  BaseArray fx (size.first, size.second);
  BaseArray fy (size.first, size.second);
  BaseArray ft (size.first, size.second);
  BaseArray fxx (size.first, size.second);
  BaseArray fxy (size.first, size.second);
  BaseArray fyy (size.first, size.second);
  BaseArray fxt (size.first, size.second);
  BaseArray fyt (size.first, size.second);


  // first order derivatives
  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      fx[i][j] = 0.5 * (i1(i+1, j, true) - i1(i-1, j, true) + i2(i+1, j, true) - i2(i-1, j, true))/(2.0 * hx);
      fy[i][j] = 0.5 * (i1(i, j+1, true) - i1(i, j-1, true) + i2(i, j+1, true) - i2(i, j-1, true))/(2.0 * hy);
      ft[i][j] = i2(i, j, true) - i1(i, j, true);
    }
  }

  // second order derivatives
  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      fxx[i][j]=(fx(i+1, j, true) - fx(i-1, j, true))/(2.0 * hx);
      fxy[i][j]=(fy(i+1, j, true) - fy(i-1, j, true))/(2.0 * hx);
      fyy[i][j]=(fy(i, j+1, true) - fy(i, j-1, true))/(2.0 * hx);
      fxt[i][j]=(ft(i+1, j, true) - ft(i-1, j, true))/(2.0 * hx);
      fyt[i][j]=(ft(i, j+1, true) - ft(i, j-1, true))/(2.0 * hx);
    }
  }

  // compute Brigthness Constancy Tensor
  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      J_11[i][j] = fxx(i, j, true) * fxx(i, j, true) + fxy(i, j, true) * fxy(i, j ,true);
      J_22[i][j] = fxy(i, j, true) * fxy(i, j, true) + fyy(i, j, true) * fyy(i, j, true);
      J_33[i][j] = fxt(i, j, true) * fxt(i, j, true) + fyt(i, j, true) * fyt(i, j, true);
      J_12[i][j] = fxx(i, j, true) * fxy(i, j, true) + fxy(i, j, true) * fyy(i, j, true);
      J_13[i][j] = fxx(i, j, true) * fxt(i, j, true) + fxy(i, j, true) * fyt(i, j, true);
      J_23[i][j] = fxy(i, j, true) * fxt(i, j, true) + fyy(i, j, true) * fyt(i, j, true);
    }
  }
  std::cout << J_11[10][10] << std::endl;
}



void HornSchunckLevelLoop(int level, int iter, double alpha, double omega, double wrapfactor, Img &i1, Img &i2, FlowField &c, FlowField &d){
    // call outer loop
    for (int i = level; i >= 0; i--){
      // calculate new size
      std::pair<int, int> sizeOriginal = i1.SizeOriginal();
      std::pair<int, int> size;
      size.first = floor(sizeOriginal.first * pow(wrapfactor, i));
      size.second = floor(sizeOriginal.second * pow(wrapfactor, i));

      std::cout << "LevelLoop: " << i << " size: " << size.first << "," << size.second << std::endl;

      HornSchunckOuterLoop(iter, alpha, omega, size, i1, i2, c, d);
    }

}

void HornSchunckOuterLoop(int iter,
                          double alpha,
                          double omega,
                          std::pair<int, int> size,
                          Img &i1,
                          Img &i2,
                          FlowField &c,
                          FlowField &d){

  // resample images
  i1.ResampleArea(size.first, size.second);
  i2.ResampleArea(size.first, size.second);
  //i1.ResampleLanczos(size.first, size.second, 3);
  //i2.ResampleLanczos(size.first, size.second, 3);

  // resample flowfields
  c.ResampleArea(size.first, size.second);
  d.ResampleArea(size.first, size.second);
  //c.ResampleLanczos(size.first, size.second, 3);
  //c.ResampleLanczos(size.first, size.second, 3);

  // set fractional flow field to zero
  d.SetAll(0);

  // compensate image 2 with flowfield
  WrapImage(i1, i2, c);

  // initialize and calculate motion tensor
  BaseArray J_11 (size.first, size.second);
  BaseArray J_22 (size.first, size.second);
  BaseArray J_33 (size.first, size.second);
  BaseArray J_12 (size.first, size.second);
  BaseArray J_13 (size.first, size.second);
  BaseArray J_23 (size.first, size.second);

  ComputeConstancyTensor(i1, i2, J_11, J_22, J_33, J_12, J_13, J_23);

  for (int i = 0; i < iter; i++){

    // change here phi for lagged nonlinearity if implemented

    // call innerloop for 1 sor step
    HornSchunckInnerLoop(alpha, omega, i1, i2, J_11, J_22, J_33, J_12, J_13, J_23, c, d);
  }

  // add fractional flowfield to current flowfield
  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      c.u[i][j] += d.u(i, j, true);
      c.v[i][j] += d.v(i, j, true);
    }
  }

}

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
                          FlowField &d){

  std::pair<int, int> size = i1.Size();
  double xp, xm, yp, ym, sum, hx, hy;

  // make sure image have same dimension
  if (size != i2.Size()){
    std::cout << "images dimensions dont match in InnerLoop" << std::endl;
    std::exit(1);
  }

  // set helper variables
  hx = alpha/(i1.Hx() * i1.Hx());
  hy = alpha/(i1.Hy() * i1.Hy());

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){

      // calculate weights
      xp =  (i < size.first - 1) * hx;
      xm =  (i > 0) * hx;
      yp =  (j < size.second - 1) * hy;
      ym =  (j > 0) * hy;
      sum = xp + xm + yp + ym;

      // calculate dv and du
      d.u[i][j] = (1-omega) * d.u(i, j, false) + omega *
                  ( (J_12(i, j, false) * d.v(i, j, true) + J_13(i, j, false)
                  - xm * ( d.u(i-1, j, false) + c.u(i-1, j, false))
                  - xp * ( d.u(i+1, j, false) + c.u(i+1, j, false))
                  - ym * ( d.u(i, j-1, false) + c.u(i, j-1, false))
                  - yp * ( d.u(i, j+1, false) + c.u(i, j+1, false))
                  + sum * ( c.u(i, j, false )))
                  / ( - J_11(i, j, false) - sum ) );

      d.v[i][j] = (1-omega) * d.v(i, j, false) + omega *
                  ( (J_12(i, j, false) * d.u(i, j, false) + J_23(i, j, false)
                  - xm * ( d.v(i-1, j, false) + c.v(i-1, j, false))
                  - xp * ( d.v(i+1, j, false) + c.v(i+1, j, false))
                  - ym * ( d.v(i, j-1, false) + c.v(i, j-1, false))
                  - yp * ( d.v(i, j+1, false) + c.v(i, j+1, false))
                  + sum * ( c.v(i, j, false )))
                  / ( - J_22(i, j, false) - sum ) );
    }
  }
}


void WrapImage(Img &i1, Img &i2, FlowField &c){
  // move the image according to the flow field
  std::pair<int, int> size = i2.Size();
  std::vector< std::vector<double> > tmp (size.first, std::vector<double>(size.second));
  double xpos, ypos;
  int xb, yb;
  double hx = 1.0/i1.Hx();
  double hy = 1.0/i1.Hy();

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){

      // test if current position + displacement is out of the area
      xpos = c.u(i, j, true) * hx + i;
      ypos = c.v(i, j, true) * hy + j;

      if (xpos < 0 || xpos >= size.first || ypos < 0 || ypos >= size.second){
        // set to i1 (later, set mask which sets dataterm in calculation to zero)
        tmp[i][j] = i1(i, j, true);
      } else {
        // calculate new pixel with bilinear interpolation
        xb = floor(xpos);
        yb = floor(ypos);

        tmp[i][j] = BilinearInterpolation(i2(xb, yb, true),
                                          i2(xb+1, yb, true),
                                          i2(xb, yb+1, true),
                                          i2(xb+1, yb+1, true),
                                          xpos-(double)xb,
                                          ypos-(double)yb);
      }
      //std::cout << i2(i, j, true) << "=" << tmp[i][j] << ", ";
    }
  }

  i2.SetTo(tmp);
}
