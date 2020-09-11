#include <iostream>
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "misc.hpp"



#define WINDOW_NAME "optical flow"



/**
  * computes a color representation of the current flow
  * @param cv::Mat f Current Flowfield
  * @return cv::Mat A RGB image of the flowfield, visualized with hsv conversion
*/



int main(int argc, char *argv[]){

  // make sure we have enough commandline arguments
  if (argc < 3){
    std::cout << "use parameters: filename1, filename2, (savefilename), (filenametruth)" << std::endl;;
    std::exit(1);
  }

  // get the filenames
  std::string filename1 (argv[1]);
  std::string filename2 (argv[2]);
  std::string truthfilename;
  std::string savefilename;
  if (argc > 3){
    savefilename = argv[3];
  } else {
    savefilename = "";
  }

  if (argc > 4){
    truthfilename = argv[4];
  } else {
    truthfilename = "";
  }

  // load the images, and make sure the exist
  cv::Mat image1 = cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
  if (image1.empty()){
    std::cout << "image 1 not found" << std::endl;
    std::exit(1);
  }

  cv::Mat image2 = cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
  if (image2.empty()){
    std::cout << "image2 not found" << std::endl;
    std::exit(1);
  }

  // load truthfile
  GroundTruth truth(truthfilename);
  
  // create window
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // parameters for computation
  int keyCode = 0;
  std::unordered_map<std::string, parameter> parameters;
  setupParameters(parameters);


  // create flowfield
  cv::Mat_<cv::Vec2d> flowfield(image1.size());
  flowfield = cv::Vec2d(0,0);

  // create image for display and helper matrixes
  cv::Mat displayimage(image1.rows, image1.cols + flowfield.cols, CV_8UC3);

  // create trackbars for every parameter
  for (auto &i: parameters){
    cv::createTrackbar(i.first, WINDOW_NAME, &i.second.value, i.second.maxvalue, TrackbarCallback, static_cast<void*>(&i.second));
  }

  cv::Mat error;
  // main loop which recomputes the optical flow with the new parameters
  while(true){

    // copy image1 into displayimage
    cv::Mat left(displayimage, cv::Rect(0, 0, image1.cols, image1.rows));
    cv::Mat right(displayimage, cv::Rect(image1.cols, 0, image1.cols, image1.rows));
    cv::cvtColor(image1, left, CV_GRAY2RGB);
    right = right * 0;
    computeColorFlowField(flowfield, right);
    computeColorFlowFieldError(flowfield, truth, error);
    cv::imshow("error", error);

    cv::imshow(WINDOW_NAME, displayimage);

    keyCode = cv::waitKey();

    // quit on ESC Key
    if (keyCode == 27){
      std::exit(0);
    }

    // recompute on Enter Key
    if (keyCode == 13){
      computeFlowField(image1, image2, parameters, flowfield);

      std::cout << std::endl << "recomputed flow with:" << std::endl;
      for (auto i: parameters){
        std::cout << i.first << ": " << std::floor((double)i.second.value*100/i.second.divfactor)/100 << std::endl;
      }

      if (truthfilename != ""){
        std::cout << std::endl << "AAE: " << truth.computeAngularError(flowfield) << std::endl;
        std::cout << "EPE: " << truth.computeEndpointError(flowfield) << std::endl;
      }
    }

    // save on s key
    if (keyCode == 115 && savefilename != ""){
      cv::FileStorage f(savefilename, cv::FileStorage::WRITE);
      f << "flowfield" << flowfield;
      f.release();
      std::cout << "flowfield was written to file: " << savefilename << std::endl;
    }

  }
}
