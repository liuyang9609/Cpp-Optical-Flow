#include <iostream>
#include "image_class.h"
#include "flow_utility.h"
#include "filehandling.h"
#include "lodepng.h"
#include <utility>
#include "hornschunck.h"

int main(int argc, char *argv[]){

  if (argc < 10){
    std::cout << "use following command line arguments" << std::endl;
    std::cout << "img1 img2 truth numlevel alpha wrapfactor omega maxiter sigma" << std::endl;
    std::exit(1);
  }
  std::string filename1 (argv[1]);
  std::string filename2 (argv[2]);
  std::string truthfilename (argv[3]);
  int level = std::atoi(argv[4]);
  double alpha = std::atof(argv[5]);
  double wrapfactor = std::atof(argv[6]);
  double omega = std::atof(argv[7]);
  int maxiter = std::atoi(argv[8]);
  double sigma = std::atof(argv[9]);

  std::cout << filename1 << std::endl;
  std::cout << filename2 << std::endl;
  std::cout << level << std::endl;
  std::cout << alpha << std::endl;
  std::cout << wrapfactor << std::endl;
  std::cout << omega << std::endl;
  std::cout << maxiter << std::endl;
  std::cout << sigma << std::endl;

  Img image1;
  Img image2;
  FlowField c;
  FlowField d;
  FlowField truth;

  // load files
  loadPGMImage(filename1, image1);
  loadPGMImage(filename2, image2);
  //loadBarronFile(truthfilename, truth);

  // resize flowfields
  c.Resize(image1.Size());
  d.Resize(image2.Size());

  // make sure image 1 and image 2 have same size
  std::pair<int, int> size = image1.Size();
  if (size != image2.Size()){
    std::cout << "Dimension of images are not equal" << std::endl;
    std::exit(1);
  }

  image1.GaussianSmoothOriginal(sigma);
  image1.SetTo(image1.getOriginal());
  writePNGImage("test.png", image1);

  image2.GaussianSmoothOriginal(sigma);

  HornSchunckLevelLoop(level, maxiter, alpha, omega, wrapfactor, image1, image2, c, d);

  c.writeToPNG("flowfield.png");
  //c.writeErrorToPNG("flowfield-error.png", truth);
  //std::cout << c.CalcAngularError(truth) << std::endl;

}
