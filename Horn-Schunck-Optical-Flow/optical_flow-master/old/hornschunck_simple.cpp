#include "hornschunck_simple.h"


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
}

void SORiteration(Img &i1,
                  Img &i2,
                  double alpha,
                  double omega,
                  int maxiter,
                  FlowField &c){

  // determine size
  std::pair<int, int> size = i1.Size();

  // compute Constancy Tensor
  BaseArray J_11 (size.first, size.second);
  BaseArray J_22 (size.first, size.second);
  BaseArray J_33 (size.first, size.second);
  BaseArray J_12 (size.first, size.second);
  BaseArray J_13 (size.first, size.second);
  BaseArray J_23 (size.first, size.second);

  ComputeConstancyTensor(i1, i2, J_11, J_22, J_33, J_12, J_13, J_23);

  // main loop
  for (int i = 0; i < maxiter; i++){
    SORiterationStep(i1, i2, alpha, omega, J_11, J_22, J_33, J_12, J_13, J_23, c);
  }
}


void SORiterationStep(Img &i1,
                      Img &i2,
                      double alpha,
                      double omega,
                      BaseArray &J_11,
                      BaseArray &J_22,
                      BaseArray &J_33,
                      BaseArray &J_12,
                      BaseArray &J_13,
                      BaseArray &J_23,
                      FlowField &c){

  // set helper variables
  double hx, hy, xp, xm, yp, ym, sum;
  hx = alpha/(i1.Hx() * i1.Hx());
  hy = alpha/(i1.Hy() * i1.Hy());
  std::pair<int, int> size = i1.Size();


  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      // calculate weights
      xp =  (i < size.first - 1) * hx;
      xm =  (i > 0) * hx;
      yp =  (j < size.second - 1) * hy;
      ym =  (j > 0) * hy;
      sum = xp + xm + yp + ym;

      // horn schunck step
      c.u[i][j] = (1 - omega) * c.u(i, j, true) + omega *
        (
          - J_13(i,j, true) - J_12(i, j, true) * c.v(i, j, false)
          + xp * c.u(i+1, j, false)
          + xm * c.u(i-1, j, false)
          + yp * c.u(i, j+1, false)
          + ym * c.u(i, j-1, false)
        )/(J_11(i, j, true) + sum);

      c.v[i][j] = (1 - omega) * c.v(i, j, true) + omega *
        (
          - J_23(i,j, true) - J_12(i, j, true) * c.u(i, j, false)
          + xp * c.v(i+1,  j, false)
          + xm * c.v(i-1, j, false)
          + yp * c.v(i, j+1, false)
          + ym * c.v(i, j-1, false)
        )/(J_22(i, j, true) + sum);

    }
  }
}

// helper functions
double H(double x, double h){
  return 0.5 * (1.0 + (2.0/M_PI) * std::atan(x/h));
}

double Hdot(double x, double h){
  return 1.0/M_PI * h/(h * h + x * x);
}

double L1(double x){
  if (x <= 0){
    return 0;
  }

  double epsilon = 0.01;
  return std::sqrt(x + epsilon);
}

double L1dot(double x){
  x = (x <= 0) ? 0 : x;
  double epsilon = 0.01;
  return 1.0 / (2 * std::sqrt(x + epsilon));
}
/*
void SORiterationStepSeparation(Img &i1,
                                Img &i2,
                                double alpha,
                                double omega,
                                double kappa,
                                double deltat,
                                BaseArray &J_11,
                                BaseArray &J_22,
                                BaseArray &J_33,
                                BaseArray &J_12,
                                BaseArray &J_13,
                                BaseArray &J_23,
                                FlowField &p,
                                FlowField &m,
                                BaseArray &phi){

  // helper variables
  double hx, hy, xp, xm, yp, ym, sum;
  hx = i1.Hx();
  hy = i1.Hy();
  std::pair<int, int> size = i1.Size();

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){

      p.u[i,j] = (1 - omega) * p.u(i,j,true);
      p.u[i,j] += omega * horn_schunck_separation_u(p, i, j, J_11, J_22, J_33, J_12, J_13, J_23, phi, alpha, kappa, hx, hy, std::pair<int,int> size);

      p.v[i,j] = (1 - omega) * p.v(i,j,true);
      p.v[i,j] += omega * horn_schunck_separation_v(p, i, j, J_11, J_22, J_33, J_12, J_13, J_23, phi, alpha, kappa, hx, hy, std::pair<int,int> size);

      m.u[i,j] = (1 - omega) * p.u(i,j,true);
      m.u[i,j] += omega * horn_schunck_separation_u(m, i, j, J_11, J_22, J_33, J_12, J_13, J_23, phi, alpha, kappa, hx, hy, std::pair<int,int> size);

      m.v[i,j] = (1 - omega) * p.v(i,j,true);
      m.v[i,j] += omega * horn_schunck_separation_v(m, i, j, J_11, J_22, J_33, J_12, J_13, J_23, phi, alpha, kappa, hx, hy, std::pair<int,int> size);

      phi[i,j] = (1 - omega) * phi(i,j,true);
      phi[i,j] += omega * horn_schunck_separation_phi(phi, p, m, i, j, J_11, J_22, J_33, J_12, J_13, J_23, phi, alpha, kappa, deltat, hx, hy, std::pair<int,int> size);
    }
  }

}


// the update of u for the simple horn-schunck with separtion
double horn_schunck_separation_u(FlowField &f,
                                 int i,
                                 int j,
                                 BaseArray &J_11,
                                 BaseArray &J_22,
                                 BaseArray &J_33,
                                 BaseArray &J_12,
                                 BaseArray &J_13,
                                 BaseArray &J_23,
                                 BaseArray &phi,
                                 double alpha,
                                 double kappa,
                                 double hx,
                                 double hy
                                 std::pair<int, int> size){
  // declare helper variables
  double n, phi, xp, xm, yp, ym;
  n = 0;
  p = phi(i,j,true);

  // check if phi is positiv for current position
  if ( p > 0 ){

    // calculate stencil
    xp =  (i < size.first - 1) * (1.0/(hx*hx));
    xm =  (i > 0) * (1.0/(hx*hx));
    yp =  (j < size.second - 1) * (1.0/(hy*hy));
    ym =  (j > 0) * (1.0/(hy*hy));
    sum = xp + xm + yp + ym;

    // TODO: add kappa
    // constancy terms
    n = J_12(i,j,true) * f.v(i,j,true) + J_13(i,j,true);

    // smoothness terms
    n += - alpha * ( xp * f.u(i+1,j,true) + xm * f.u(i-1,j,true) + yp * f.u(i,j+1,true) + ym * f.u(i,j-1,true));
    n += - alpha * H(p)/Hdot(p) * ( ( phi(i+1,j,true) - phi(i-1,j,true) )/(2*hx) + ( phi(i,j+1,true) - phi(i,j-1,true) )/(2*hy) );

    // normalization terms
    n /= (-J_11(i,j,true) - alpha * sum);

  } else {
    // just use smoothness term as defined in paper
    n = (f.u(i+1,j,true) + f.u(i-1,j,true) + f.u(i,j+1,true) - f.u(i,j-1,true))/4.0;
  }

  return n;
}


// the update of v for the simple horn-schunck with separtion
double horn_schunck_separation_v(FlowField &f,
                                 int i,
                                 int j,
                                 BaseArray &J_11,
                                 BaseArray &J_22,
                                 BaseArray &J_33,
                                 BaseArray &J_12,
                                 BaseArray &J_13,
                                 BaseArray &J_23,
                                 BaseArray &phi,
                                 double alpha,
                                 double kappa,
                                 double hx,
                                 double hy
                                 std::pair<int, int> size){
  // declare helper variables
  double n, phi, xp, xm, yp, ym;
  n = 0;
  p = phi(i,j,true);

  // check if phi is positiv for current position
  if ( p > 0 ){

    // calculate stencil
    xp =  (i < size.first - 1) * (1.0/(hx*hx));
    xm =  (i > 0) * (1.0/(hx*hx));
    yp =  (j < size.second - 1) * (1.0/(hy*hy));
    ym =  (j > 0) * (1.0/(hy*hy));
    sum = xp + xm + yp + ym;

    // TODO: add kappa
    // constancy terms
    n = J_12(i,j,true) * f.u(i,j,true) + J_23(i,j,true);

    // smoothness terms
    n += - alpha * ( xp * f.v(i+1,j,true) + xm * f.v(i-1,j,true) + yp * f.v(i,j+1,true) + ym * f.v(i,j-1,true));
    n += - alpha * H(p)/Hdot(p) * ( ( phi(i+1,j,true) - phi(i-1,j,true) )/(2*hx) + ( phi(i,j+1,true) - phi(i,j-1,true) )/(2*hy) );

    // normalization terms
    n /= (-J_22(i,j,true) - alpha * sum);

  } else {
    // just use smoothness term as defined in paper
    n = (f.v(i+1,j,true) + f.v(i-1,j,true) + f.v(i,j+1,true) - f.v(i,j-1,true))/4.0;
  }

  return n;
}


// update the separation field
double horn_schunck_separation_phi(BaseArray &phi,
                                   FlowField &p,
                                   FlowField &m,
                                   int i,
                                   int j,
                                   BaseArray &J_11,
                                   BaseArray &J_22,
                                   BaseArray &J_33,
                                   BaseArray &J_12,
                                   BaseArray &J_13,
                                   BaseArray &J_23,
                                   double alpha,
                                   double kappa,
                                   double deltat,
                                   double hx,
                                   double hy
                                   std::pair<int, int> size){

  // helper variables
  double n, ph, hdot, phix, phiy, phixx, phiyy, phixy;

  // compute helper variables
  hdot = Hdot(phi(i,j,true));
  phix = (phi(i+1,j,true) - phi(i-1,j,true))/(2 * hx);
  phiy = (phi(i,j+1,true) - phi(i,j-1,true))/(2 * hy);

  // extra test on boundary for 2nd order derivatives
  phixx = (i < size.first -1) * (phi(i+1,j,true) - phi(i,j,true))/(hx*hx) + (i > 0) * (phi(i-1,j,true) - phi(i,j,true))/(hx*hx);
  phiyy = (j < size.second -1) * (phi(i,j+1,true) - phi(i,j,true))/(hy*hy) + (j > 0) * (phi(i,j-1,true) - phi(i,j,true))/(hy*hy);
  phixy = (i > 0) * (i < size.first -1) * (j > 0) * (j < size.second -1) *
          (phi(i+1,j+1, true) + phi(i-1,j-1,true) - phi(i-1,j+1,true) - phi(i+1,j-1,true))/(hx*hy);

  ph = phi(i,j,true);


  // adding the positive smoothness terms
  n =  - alpha * hdot * (p.u(i+1,j,true) - p.u(i-1,j,true) + p.v(i+1,j,true) - p.v(i-1,j,true))/(2*hx);
  n += - alpha * hdot * (p.u(i,j+1,true) - p.u(i,j-1,true) + p.v(i,j+1,true) - p.v(i,j-1,true))/(2*hy);

  // adding the negative smoothness terms
  n += alpha * hdot * (m.u(i+1,j,true) - m.u(i-1,j,true) + m.v(i+1,j,true) - m.v(i-1,j,true))/(2*hx);
  n += alpha * hdot * (m.u(i,j+1,true) - m.u(i,j-1,true) + m.v(i,j+1,true) - m.v(i,j-1,true))/(2*hy);

  // adding the postive data terms
  n += - kappa * hdot * (    J_11(i,j,true) * p.u(i,j,true) * p.u(i,j,true) +
                             J_22(i,j,true) * p.v(i,j,true) * p.v(i,j,true) +
                             J_33(i,j,true) +
                         2 * J_12(i,j,true) * p.u(i,j,true) * p.v(i,j,true) +
                         2 * J_13(i,j,true) * p.u(i,j,true) +
                         2 * J_23(i,j,true) * p.v(i,j,true));

  // adding the negative data terms
  n += kappa * hdot * (    J_11(i,j,true) * m.u(i,j,true) * m.u(i,j,true) +
                           J_22(i,j,true) * m.v(i,j,true) * m.v(i,j,true) +
                           J_33(i,j,true) +
                       2 * J_12(i,j,true) * m.u(i,j,true) * m.v(i,j,true) +
                       2 * J_13(i,j,true) * m.u(i,j,true) +
                       2 * J_23(i,j,true) * m.v(i,j,true));

  // last term...
  phi_abs = std::sqrt(phix * phix + phiy * phiy);
  n += hdot * (phix * phix *phiyy + phiy * phiy * phixx - 2 * phix * phiy * phixy)/pow(phi_abs, 3);

  return ph + deltat * n;
}
*/
