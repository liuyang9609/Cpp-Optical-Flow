#pragma once

#include "ofMain.h"

#include "ofxOpenCv.h"
#include "ofxOpticalFlowBM.h"
#include "ofxQtVideoSaver.h"


//#define _USE_LIVE_VIDEO		// uncomment this to use a live camera
								// otherwise, we'll use a movie file

class testApp : public ofBaseApp{

	public:
    ~testApp();
    
    void createFileName();
  
		void setup();
		void update();
		void draw();		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
  
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);	
	
    ofxOpticalFlowBM flow;
    ofxQtVideoSaver videoSaver;
    
    string filename;
    ofImage screen;
    float frameCount;
   
    bool screenReady;
    
    unsigned char * currPixels;
    unsigned char * prevPixels;
      
    ofVideoPlayer video;

};

