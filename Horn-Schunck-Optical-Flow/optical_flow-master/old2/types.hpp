#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define TAG_FLOAT 202021.25
#define UNKNOWN_FLOW_THRESH 1e9


struct parameter {
  std::string name;
  int value;
  int maxvalue;
  int divfactor;
};

struct tensor {
  double J11;
  double J22;
  double J33;
  double J12;
  double J13;
  double J23;
};



template<> class cv::DataType<tensor> {
public:
  typedef tensor channel_type;
  enum {
    channels = 6,
    type=CV_MAKETYPE(64, 6)
  };
};



class GroundTruth {

 public:
  cv::Mat_<cv::Vec2d> truthfield;
  cv::Mat_<int> mask;

  GroundTruth(std::string filename){
    
    // test if filename is set
    if (filename == ""){
      std::cout << "now filename given in ground truth " << std::endl;
      return;
    }

    // which fileformat to load?
    unsigned dot_found = filename.find_last_of(".");
    std::string fileextension = filename.substr(dot_found, filename.length()-dot_found);
    
    if (fileextension == ".flo") {
      loadFlowFile(filename);
    } else if (fileextension == ".png") {
      loadKittiFile(filename);
    } else if (fileextension == ".F") {
      loadBarrenFile(filename);
    } else {
      std::cout << "unkown file format for ground truth" << std::endl;
      return;
    }
  }



  double computeAngularError(cv::Mat_<cv::Vec2d> flowfield) {
    double amount = 0;
    double tmp1 = 0;
    double sum_ang = 0;

    for (int i = 0; i < flowfield.rows; i++) {
      for (int j = 0; j < flowfield.cols; j++){
        // test if reference flow vector exists
        if (mask(i,j) == 1){
          amount++;
          tmp1 = (truthfield(i,j)[0] * flowfield(i,j)[0] + truthfield(i,j)[1] * flowfield(i,j)[1] + 1.0)
                  / std::sqrt( (truthfield(i,j)[0] * truthfield(i,j)[0] + truthfield(i,j)[1] * truthfield(i,j)[1] + 1.0) *
                          (flowfield(i,j)[0] * flowfield(i,j)[0] + flowfield(i,j)[1] * flowfield(i,j)[1] + 1.0) );

          if (tmp1 > 1.0) tmp1 = 1.0;
          if (tmp1 < - 1.0) tmp1 = - 1.0;
          sum_ang += acos(tmp1) * 180.0/M_PI;
        }
      }
    }
    return sum_ang / amount;
  }



  double computeEndpointError(cv::Mat_<cv::Vec2d> flowfield) {
    // compute endpoint error
    
    double amount = 0;
    double sum = 0;
    double diff_u = 0;
    double diff_v = 0;

    for (int i = 0; i < flowfield.rows; i++){
      for (int j = 0; j < flowfield.cols; j++){
        if (mask(i,j) == 1){
          diff_u = flowfield(i,j)[0] - truthfield(i,j)[0];
          diff_v = flowfield(i,j)[1] - truthfield(i,j)[1];

          sum += std::sqrt(diff_u * diff_u + diff_v * diff_v);
          amount++;
        }
      }
    }
    return sum/amount;
  }


 private:
  bool isSet = false;
  

  void loadBarrenFile(std::string filename){
    // load the ground truth specified in the Barron format (i.e yosemite sequence)

    FILE *file = fopen(filename.c_str(), "r");
    if (file == NULL){
      std::cout << "Error opening Barron File: " << filename << "! Exit" << std::endl;
      std::exit(1);
    }

    // read the header information
    float help;
    std::fread (&help, sizeof(float), 1, file);
    int nx_and_offsetx  = (int) help;
    std::fread (&help, sizeof(float), 1, file);
    int ny_and_offsety  = (int) help;
    std::fread (&help, sizeof(float), 1, file);
    int nx  = (int) help;
    std::fread (&help, sizeof(float), 1, file);
    int ny  = (int) help;
    std::fread (&help, sizeof(float), 1, file);
    int offsetx = (int) help;
    std::fread (&help, sizeof(float), 1, file);
    int offsety = (int) help;

    // initialize truth vector field
    truthfield.create(ny, nx);
    mask.create(ny, nx);

    // make tmp array
    std::vector< std::vector<double> > tmpu(nx_and_offsetx, std::vector<double>(ny_and_offsety));
    std::vector< std::vector<double> > tmpv(nx_and_offsetx, std::vector<double>(ny_and_offsety));

    // read complete data
    for (int j = 0; j < ny_and_offsety; j++){
      for (int i = 0; i < nx_and_offsetx; i++){
        fread(&help, sizeof(float), 1, file);
        tmpu[i][j] = (double)help;

        fread(&help, sizeof(float), 1, file);
        tmpv[i][j] = (double)help;
      }
    }
    fclose(file);

    // set data without crop
    for (int i = 0; i < ny; i++){
      for (int j = 0; j < nx; j++){
        truthfield(i,j)[0] = tmpu[j + offsetx][i + offsety];
        truthfield(i,j)[1] = tmpv[j + offsetx][i + offsety];

        // if both values are 100.0 the flowfield at that point is not valid
        if (truthfield(i,j)[0] == 100.0 && truthfield(i,j)[1] == 100.0){
          mask(i,j) = 0;
        } else {
          mask(i,j) = 1;
        }
      }
    }

    isSet = true;
  }



  // used for middlebury sequences
  void loadFlowFile(std::string filename){
    if (filename == "") {
	std::cout << "ReadFlowFile: empty filename" << std::endl;
        std::exit(1);
    }

    FILE *stream = fopen(filename.c_str(), "rb");
    if (stream == 0) {
        std::cout << "ReadFlowFile: could not open " << filename << std::endl;
        std::exit(1);
    }
    
    int width, height;
    float tag;

    if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
	(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
	(int)fread(&height, sizeof(int),   1, stream) != 1) {
	std::cout << "ReadFlowFile: problem reading file " << filename << std::endl;
        std::exit(1);
    }

    if (tag != TAG_FLOAT) { // simple test for correct endian-ness
	std::cout << "ReadFlowFile: wrong tag (possibly due to big-endian machine?) " << filename << std::endl;
        std::exit(1);
    }

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999) {
	std::cout << "ReadFlowFile: illegal width " << width << " with file " << filename << std::endl;
        std::exit(1);
    }

    if (height < 1 || height > 99999) {
	std::cout << "ReadFlowFile: illegal height " << height << " with file " << filename << std::endl;
        std::exit(1);
    }

    truthfield.create(height, width);
    mask.create(height, width);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    float value = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if ((int)fread(&value, sizeof(float), 1, stream) != 1) {
          std::cout << "file is too short " << filename << std::endl;
          std::exit(1);
        }
        truthfield(y,x)[0] = (is_valid_middleburry_flow(value)) ? value : 0;
        
        if ((int)fread(&value, sizeof(float), 1, stream) != 1) {
          std::cout << "file is too short " << filename << std::endl;
          std::exit(1);
        }
        truthfield(y,x)[1] = (is_valid_middleburry_flow(value)) ? value : 0;

        // set mask if flow is valid
        mask(y,x) = (is_valid_middleburry_flow(truthfield(y,x)[0]) && is_valid_middleburry_flow(truthfield(y,x)[1])) ? 1 : 0;
      }
    }

    if (fgetc(stream) != EOF) {
	std::cout << "ReadFlowFile: file is too long " << filename << std::endl;
        std::exit(1);
    }
    
    fclose(stream);
    isSet = true;
  }

  double is_valid_middleburry_flow(double value){
    return std::abs(value) < UNKNOWN_FLOW_THRESH;
  }



  void loadKittiFile(std::string filename){
    // load the ground truth provided by the kitti benchmark

    cv::Mat file = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (file.data == NULL){
      std::cout << "could not open ground truth file: " << filename << std::endl;
      std::exit(1);
    }
    mask.create(file.size());
    truthfield.create(file.size());


    for (int i = 0; i < file.rows; i++){
      for (int j = 0; j < file.cols; j++){
        if (file.at<cv::Vec3s>(i,j)[0] == 0){
          truthfield(i,j)[0] = 0;
          truthfield(i,j)[1] = 0;
          mask(i,j) = 0;
        } else {
          truthfield(i,j)[0] = (double)((float)(unsigned short)file.at<cv::Vec3s>(i,j)[2] - 32768)/64.0;
          truthfield(i,j)[1] = (double)((float)(unsigned short)file.at<cv::Vec3s>(i,j)[1] - 32768)/64.0;
          mask(i,j) = 1;
        }
      }
    }
    isSet = true;
  }

};


#endif
