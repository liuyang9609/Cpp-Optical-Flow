#include "initial_separation.hpp"


void initial_segmentation(const cv::Mat_<cv::Vec2d> &flowfield,
                          //const cv::Mat_<cv::Vec2d> &initialflowfield
                          cv::Mat_<double> &phi,
                          const std::unordered_map<std::string, parameter> &parameters,
                          cv::Vec6d &dominantmotion
                        ){

  segementFlowfield(flowfield, phi, parameters, dominantmotion);

  // box median filter of size 7x7 to get smooth edges on segement borders
  cv::blur(phi, phi, cv::Size(7,7), cv::Point(-1, -1), cv::BORDER_REPLICATE);
  
}


void segementFlowfield(const cv::Mat_<cv::Vec2d> &f, cv::Mat_<double> &phi, const std::unordered_map<std::string, parameter> &parameters, cv::Vec6d &dominantmotion){

  // parameters for segmentation
  int blocksize = parameters.at("blocksize").value;
  double Tr = (double)parameters.at("Tr").value/parameters.at("Tr").divfactor;
  double Tm = (double)parameters.at("Tm").value/parameters.at("Tm").divfactor;
  double Ta= (double)parameters.at("Ta").value/parameters.at("Ta").divfactor;

  // number of blocks in x and y direction 
  int num_x = std::ceil((float)f.cols/blocksize), num_y = std::ceil((float)f.rows/blocksize);

  // affine parameters for each block, and a temporary mat for the least square estimation
  cv::Mat_<cv::Vec6d> affine_parameters(num_y, num_x);
  cv::Mat affine_tmp;
  
  // flags to indicate which blocks have valid affine parameters
  cv::Mat_<bool> valid(num_y, num_x);

  // matrices for the least square estimation
  // coords contain the coordinates of the block, flow_block contains the flow values inside the block
  cv::Mat_<double> coords(blocksize*blocksize, 3), flow_block(blocksize*blocksize, 2);

  // helper variables to determine the length of the block
  double xlength, ylength;
  
  // estimated affine parameters for each block
  for (int i = 0; i < num_y; i++){
    for (int j = 0; j < num_x; j++){
      xlength = ((j+1) * blocksize < f.cols) ? (j+1) * blocksize : f.cols;
      ylength = ((i+1) * blocksize < f.rows) ? (i+1) * blocksize : f.rows;

      
      /* block coordinates:
      * ( x1/y1 x2/y2 x3/y3 x4/y4
      *   x5/y5  ...
      *   ...                 ... )
      *
      * structure of coords block: 
      * coords = ( x1 y1 1
      *            x2 y2 1
      *            x3 y3 1
      *            ...     )
      *
      * structure of flow block:
      * flow_block = (u1 v1
      *               u2 v2
      *               ...  )
      *
      * least square problem:
      * (coords * affine_tmp - flowblock)^2
      */
      int k = 0;  // easiest solution TODO: change later maybe
      for (int y = i*blocksize; y < ylength; y++){
        for (int x = j*blocksize; x < xlength; x++){
          flow_block[k][0] = f(y,x)[0];
          flow_block[k][1] = f(y,x)[1];
          coords[k][0] = x;
          coords[k][1] = y;
          coords[k][2] = 1;
          k++;
        }
      }

      
      cv::solve(coords, flow_block, affine_tmp, cv::DECOMP_QR);
      affine_parameters(i,j) = affine_tmp.reshape(0,1).clone();

      // test if affine parameters are a "good" estimation
      double error = 0;
      error = error_block(i, j, blocksize, affine_parameters(i,j), f);
      valid(i,j) = error < Tr; 
    }
  }

  // merging procedure
  
  // for each valid block find all blocks, which can merge and store the maximal number of merged blocks as well as the correspondend affine motion
  int maximum_number_merged = 0;
  int number_merged = 0;
  
  cv::Vec6d best_affine, affine_pass;

  cv::Mat_<bool> merged_blocks_pass(num_y, num_x), merged_blocks_best(num_y, num_x);
  merged_blocks_pass = false;

 
  // loop through the starting blocks
  for (int i = 0; i < num_y; i++ ) {
    for (int j = 0; j < num_x; j++) {

      std::cout << "looping through starting blocks " << i << "," << j << std::endl;
      
      // choose valid blocks as starting blocks
      if (valid(i,j)) {
        
        // set the current affine parameters to the start block
        affine_pass = affine_parameters(i,j);
        merged_blocks_pass = false;

        // loop through all blocks and find merging blocks
        for (int k = 0; k < num_y; k++) {
          for (int l = 0; l < num_x; l++) {
            if (valid(k,l) && are_close_blocks(affine_parameters(i,j), affine_parameters(k,l), Tm, 1)) {
              choose_better_affine(merged_blocks_pass, affine_parameters(k,l), affine_pass, f, blocksize);
              number_merged++;
              merged_blocks_pass(k,l) = true;
            }
          }
        }
        
        // have we merged more blocks than in the previous passes?
        if (number_merged > maximum_number_merged) {
          maximum_number_merged = number_merged;
          best_affine = affine_pass;
          merged_blocks_best = merged_blocks_pass.clone();
        }

      }
    }
  }
  
  // set dominant motion
  dominantmotion = cv::Mat(best_affine).clone();

  // label each vector, if the error to the affine motion is smaller to Ta
  double error = 0;

  for (int i = 0; i < f.rows; i++) {
    for (int j = 0; j < f.cols; j++) {
      error  = std::pow(best_affine[0] * j + best_affine[2] * i + best_affine[4] - f(i,j)[0], 2);
      error += std::pow(best_affine[1] * j + best_affine[3] * i + best_affine[5] - f(i,j)[1], 2);

      phi(i,j) = (error < Ta) ? 1 : -1;
    }
  }
}





bool are_close_blocks(cv::Vec6d a1, cv::Vec6d a2, double Tm, double r){
  // distance function from wang
  cv::Vec6d diff = a1-a2;
  double value = diff[0] * diff[0] * 1 +
                 diff[1] * diff[1] * r * r +
                 diff[2] * diff[2] * r * r +
                 diff[3] * diff[3] * 1 +
                 diff[4] * diff[4] * r * r +
                 diff[5] * diff[5] * r * r;
  return std::sqrt(value) < Tm;
}





void choose_better_affine(
    const cv::Mat_<bool> &merged_blocks,
    const cv::Vec6d &p_new,
    cv::Vec6d &p_old,
    const cv::Mat_<cv::Vec2d> &f,
    int blocksize
  ){
  
  double error_new = 0;
  double error_old = 0;

  // loop over all blocks and compute the error for p_new and for p_old for all blocks which have been merged
  for (int i = 0; i < merged_blocks.rows; i++) {
    for (int j = 0; j < merged_blocks.cols; j++) {
      if (merged_blocks(i,j)) {
        error_new += error_block(i, j, blocksize, p_new, f);
        error_old += error_block(i, j, blocksize, p_old, f);
      }
    }
  }
  
  p_old = (error_new > error_old) ? p_old : p_new;
}



double error_block(int i, int j, int blocksize, const cv::Vec6d &a_p, const cv::Mat_<cv::Vec2d> &f){
  
  int xlength = ((j+1) * blocksize < f.cols) ? (j+1) * blocksize : f.cols;
  int ylength = ((i+1) * blocksize < f.rows) ? (i+1) * blocksize : f.rows;
  double error = 0;

  for (int y = i*blocksize; y < ylength; y++){
    for (int x = j*blocksize; x < xlength; x++){
      error += std::pow(x * a_p[0] + y * a_p[2] + a_p[4] - f(y,x)[0], 2);
      error += std::pow(x * a_p[1] + y * a_p[3] + a_p[5] - f(y,x)[1], 2);
    }
  }

  return std::sqrt(error/((xlength-j*blocksize) * (ylength-i*blocksize)));
}
