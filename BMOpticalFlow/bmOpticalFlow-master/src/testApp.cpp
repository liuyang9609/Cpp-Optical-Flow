#include "testApp.h"

//--------------------------------------------------------------

testApp::~testApp() {
  videoSaver.finishMovie();
}

//--------------------------------------------------------------

void testApp::createFileName(void)
{
  // create a uniqe file name
	ostringstream oss;
	oss << ofGetYear() << "-";
	oss << setw(2) << setfill('0') << ofGetMonth() << "-";
	oss << setw(2) << setfill('0') << ofGetDay() << "-";
	oss << setw(2) << setfill('0') << ofGetHours() << "-";
	oss << setw(2) << setfill('0') << ofGetMinutes() << "-";
	oss << setw(2) << setfill('0') << ofGetSeconds() << ".mov";
	filename = oss.str();	
}

//--------------------------------------------------------------

void testApp::setup(){
  ofSetBackgroundAuto(false); 
  //ofEnableAlphaBlending();
  ofBackground(0, 0, 0);
  
  screenReady = false;
  
  video.loadMovie("westminster_st.mov");
  //video.loadMovie("fingers.mov");
  video.play();
  video.setPosition(0.11);
  video.setPaused(true);

  flow.setup(video.width, video.height, 5, 5, 15);
  
  currPixels = video.getPixels();   
  flow.update(currPixels, video.width, video.height, OF_IMAGE_COLOR);
  
  ofNoFill();
  ofSetColor(255);  
  
  createFileName();
  frameCount = 0;
  videoSaver.setCodecQualityLevel(OF_QT_SAVER_CODEC_QUALITY_NORMAL);
  videoSaver.setup(ofGetWidth(), ofGetHeight(), filename);
  
}

//--------------------------------------------------------------
void testApp::update(){
  video.idleMovie();
  
  if( video.isFrameNew() ) {  
    prevPixels = currPixels;
    currPixels = video.getPixels();   
    flow.update(currPixels, video.width, video.height, OF_IMAGE_COLOR);
  }
  if( screenReady ) {
    screen.grabScreen(0, 0,ofGetWidth(), ofGetHeight());
    videoSaver.addFrame(screen.getPixels(), video.getPosition());
    frameCount++;
    video.nextFrame();
    screenReady = false;
  }
}

//--------------------------------------------------------------
void testApp::draw(){
  ofPoint currentVel;
  if( !screenReady ) {
  // video.draw(0, 0);
  for(int y = 0; y < video.height; y++) {
    for (int x = 0; x < video.width; x++) {
      currentVel = flow.getVel(x, y);
      ofEnableAlphaBlending();
      ofSetColor(prevPixels[((video.width * y) + x) * 3], 
                 prevPixels[((video.width * y) + x) * 3 + 1], 
                 prevPixels[((video.width * y) + x) * 3 + 2],
                 50);  //Alpha channel
      ofLine(x, y, x + currentVel.x, y + currentVel.y);
      ofDisableAlphaBlending();
    }
  }
    screenReady = true;
  } 
  ofSetColor(255, 255, 255);
  //flow.draw(0, 0);
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
  video.nextFrame();
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){
  
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}
