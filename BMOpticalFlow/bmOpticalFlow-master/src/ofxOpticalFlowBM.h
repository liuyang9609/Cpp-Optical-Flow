/*
 *  ofxOpticalFlowBM.h
 *  bmOpticalFlow
 *
 *  Created by Kelly Egan on 4/15/12.
 *
 */

#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"

class ofxOpticalFlowBM {
  public:
    ofxOpticalFlowBM();
    ~ofxOpticalFlowBM();
  
    void reset();
    void destroy();
  
    void setup( int w, int h, int blkSize = 20, int shfSize = 10 , int maxRng = 20 );
    void update( unsigned char* pixels, int width, int height, int imageType );
  
    ofPoint getBlockVel(int x, int y);    //x and y are blocks not pixels
    ofPoint getVel(int x, int y);         //x and y are pixels
    ofPoint getVelTween(int x, int y);    //Get velocity between blocks
    void draw(int xOrigin, int yOrigin);  //
    
    ofxCvColorImage		   colrImgLrg;  // full scale color image.
    ofxCvColorImage			 colrImgSml;  // scaled down color image.
  
    ofxCvGrayscaleImage  greyImgLrg;  // full scale grey image.
    ofxCvGrayscaleImage  greyImgSml;  // scaled down grey image.
    ofxCvGrayscaleImage  greyImgPrv;  // scaled down grey image - copy of previous frame.

    IplImage*				     opFlowVelX;  // optical flow in the x direction.
    IplImage*				     opFlowVelY;  // optical flow in the y direction.
    
    CvSize fullSize;
    CvSize scalSize;
    CvSize blockSize;
    CvSize shiftSize;
    CvSize maxRange;
    CvSize flowSize;
    
    bool initialized;
  
};