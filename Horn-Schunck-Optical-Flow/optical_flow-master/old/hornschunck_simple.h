#ifndef HORNSCHUNCK_SIMPLE_H
#define HORNSCHUNCK_SIMPLE_H
#include "image_class.h"


void ComputeConstancyTensor(Img &i1,
                            Img &i2,
                            BaseArray &J_11,
                            BaseArray &J_22,
                            BaseArray &J_33,
                            BaseArray &J_12,
                            BaseArray &J_13,
                            BaseArray &J_23);

void SORiteration(Img &i1,
                  Img &i2,
                  double alpha,
                  double omega,
                  int maxiter,
                  FlowField &c);


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
                      FlowField &c);

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
                                BaseArray &phi);

double H(double x, double h);
double Hdot(double x, double h);
double L1(double x);
double L1dot(double x);

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
                                 double hy,
                                 std::pair<int, int> size);

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
                                double hy,
                                std::pair<int, int> size);

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
                                   double hy,
                                   std::pair<int, int> size);

#endif
