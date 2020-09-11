#include <iostream>
#include "image_class.h"
#include "flow_utility.h"
#include "filehandling.h"
#include "lodepng.h"
#include <utility>
#include "hornschunck_simple.h"

int main(int argc, char *argv[]){
  if (argc < 8){
    std::cout << "use following command line arguments" << std::endl;
    std::cout << "img1 img2 truth alpha omega maxiter sigma" << std::endl;
    std::exit(1);
  }
  std::string filename1 (argv[1]);
  std::string filename2 (argv[2]);
  std::string truthfilename (argv[3]);
  double alpha = std::atof(argv[4]);
  double omega = std::atof(argv[5]);
  int maxiter = std::atoi(argv[6]);
  double sigma = std::atof(argv[7]);

  std::cout << filename1 << std::endl;
  std::cout << filename2 << std::endl;
  std::cout << alpha << std::endl;
  std::cout << omega << std::endl;
  std::cout << maxiter << std::endl;
  std::cout << sigma << std::endl;

  Img image1;
  Img image2;
  FlowField c;
  FlowField truth;
  BaseArray phi;

  // load files
  loadPGMImage(filename1, image1);
  loadPGMImage(filename2, image2);
  loadBarronFile(truthfilename, truth);

  // resize flowfields
  c.Resize(image1.Size());

  // make sure image 1 and image 2 have same size
  std::pair<int, int> size = image1.Size();
  if (size != image2.Size()){
    std::cout << "Dimension of images are not equal" << std::endl;
    std::exit(1);
  }

  image1.GaussianSmoothOriginal(sigma);
  image2.GaussianSmoothOriginal(sigma);

  // set images to smoothed original
  image1.SetTo(image1.getOriginal());
  image2.SetTo(image2.getOriginal());

  // initialize phi
  //UltraSimpleInitialization(phi);

  SORiteration(image1, image2, alpha, omega, maxiter, c);

  c.writeToPNG("flowfield-simple.png");
  //c.writeErrorToPNG("flowfield-error-simple.png", truth);
  std::cout << c.CalcAngularError(truth) << std::endl;

}
