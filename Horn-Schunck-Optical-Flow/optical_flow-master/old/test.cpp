#include <iostream>
#include "image_class.h"
#include "flow_utility.h"
#include "filehandling.h"
#include "lodepng.h"
#include <utility>
#include "hornschunck.h"

int main(){

  Img image1;
  Img image2;
  FlowField c;
  FlowField d;
  FlowField truth;

  // load files
  loadPNGImage("gopher.png", image1);
  image1.ResampleLanczos(image1.SizeOriginal().first * 0.125, image1.SizeOriginal().second * 0.125, 4);
  writePNGImage("goper_lanczos.png", image1);

}
