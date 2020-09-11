#include <iostream>
#include <string>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "shared/misc.hpp"
#include "shared/flow_computation.hpp"
#include "shared/types.hpp"


#define PARAMETER_WINDOW_NAME "parameter"


int main(int argc, char *argv[]){

  // make sure we have enough commandline arguments
  if (argc < 2){
    std::cout << "command line for scenario file not given" << std::endl;
    std::exit(1);
  }

  // open the scenario file
  std::string scenariofile(argv[1]);
  cv::FileStorage scenario(scenariofile, cv::FileStorage::READ);

  // get images
  cv::Mat image1, image2;
  scenario["image1"] >> image1;
  scenario["image2"] >> image2;

  // get groundtruth
  cv::Mat_<cv::Vec2d> groundtruth;
  scenario["groundtruth"] >> groundtruth;
  cv::Mat_<int> truthmask;
  scenario["truthmask"] >> truthmask;
  GroundTruth truth(groundtruth, truthmask);

  // get parameters
  std::unordered_map<std::string, parameter> parameters;
  cv::FileNode p = scenario["parameters"];
  loadParameters(p, parameters);
  
  // get interactive parameter
  bool interactive;
  scenario["interactive"] >> interactive;

  // initialize flowfield
  cv::Mat_<cv::Vec2d> flowfield(image1.size());
  flowfield = cv::Vec2d(0,0);

  // initialize segmentation 
  cv::Mat_<double> segmentation(image1.size());
  segmentation = 1;
  
  if (interactive) {
    
    int keyCode = 0;

    // create trackbars for every parameter
    cv::namedWindow(PARAMETER_WINDOW_NAME, CV_WINDOW_NORMAL);
    for (auto &i: parameters){
      cv::createTrackbar(i.first, PARAMETER_WINDOW_NAME, &i.second.value, i.second.maxvalue, TrackbarCallback, static_cast<void*>(&i.second));
    }

    while(true) {
      keyCode = cv::waitKey();
      std::cout << keyCode << std::endl;
      if (keyCode == 27){
        std::exit(0);
      }

      if (keyCode == 13) {
        computeFlowField(image1, image2, truth, segmentation, parameters, interactive, scenario);
      }
    }
  } else {
    computeFlowField(image1, image2, truth, segmentation, parameters, interactive, scenario);
  }
}

