#ifndef FILE_HANDLING_H
#define FILE_HANDLING_H

#include <string>
#include "lodepng.h"
#include <vector>
#include <iostream>
#include "image_class.h"
#include "flow_utility.h"
#include <tuple>

struct PGMHeader {
  int width;
  int height;
  int dataoffset;
};

void loadPNGImage(std::string filename, Img &dest);
void loadPGMImage(std::string filename, Img &dest);
PGMHeader readPGMHeader(std::string filename);
void writePGMImage(std::string filename, Img &src);
void writePNGImage(std::string filename, Img &src);
void writePNGImage(std::string filename, std::vector< std::vector<RGBA> > src);

void loadBarronFile(std::string filename, FlowField &dest);

#endif
