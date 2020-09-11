#include "filehandling.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>

void loadPNGImage(std::string filename, Img &dest){
  std::vector<unsigned char> raw;
  unsigned width, height;

  unsigned error = lodepng::decode(raw, width, height, filename.c_str());
  if ( error ){
    std::cout << "Error while loading File: " << filename << "! Exit" << std::endl;
    std::exit(1);
  }
  dest.Resize(width, height);
  dest.ResizeOriginal(width, height);

  for (int i = 0; i < width; i++){
    for (int j = 0; j < height; j++){
      // set color picture to black and white
      int base = j * width * 4 + i * 4;
      dest.Set(i, j, RGBtoGray(raw[base], raw[base+1], raw[base+2]));
      dest.SetOriginal(i, j, RGBtoGray(raw[base], raw[base+1], raw[base+2]));
    }
  }
}


void loadPGMImage(std::string filename, Img &dest){
  std::ifstream img;

  PGMHeader info = readPGMHeader(filename);
  dest.Resize(info.width, info.height);
  dest.ResizeOriginal(info.width, info.height);

  img.open(filename, std::ios::binary);
  if (!img.is_open()){
    std::cout << "Could not open file: " << filename << "! Exit" << std::endl;
    std::exit(1);
  }

  img.seekg(info.dataoffset);
  char memblock;
  for (int j = 0; j < info.height; j++) {
    for (int i = 0; i < info.width; i++) {
      img.read(&memblock, 1);
      dest.Set(i, j, (double)(unsigned char)memblock);
      dest.SetOriginal(i, j, (double)(unsigned char)memblock);
    }
  }
  img.close();
}


PGMHeader readPGMHeader(std::string filename){
  // assuming header information are all on new line (otherwise: fuck that shit)
  PGMHeader info;
  std::string line;

  // open file
  std::ifstream file;
  file.open(filename);
  if (!file.is_open()){
    std::cout << "Error opening the file " << filename << "! Exit" << std::endl;
    std::exit(1);
  }


  // find magic number
  while (std::getline(file, line, '\n')){
    if (line[0] == '#'){ continue;}   // ignore comments
    std::size_t found = line.find("P5");
    if (found != std::string::npos){
      break;
    }
    else {
      std::cout << "Error Magic number not found" << std::endl;
      std::exit(1);
    }
  }

  // find width and height
  std::stringstream ss;
  std::string token;
  ss.clear();
  ss.str("");
  while(std::getline(file, line,'\n')){
    if (line[0] == '#'){ continue; }    // ignore comments
    ss << line;
    ss >> token;
    info.width = std::stoi(token.c_str());
    ss >> token;
    info.height = std::stoi(token.c_str());
    break;
  }

  // ignore max value
  while(std::getline(file, line, '\n')){
    if (line[0]=='#'){continue;}    //ignore comments
    if (line !=""){break;}
  }

  std::streampos offset =  file.tellg();
  info.dataoffset = offset;
  file.close();

  return info;
}


void writePGMImage(std::string filename, Img &src){
  std::ofstream file;
  std::pair<int, int> size = src.Size();
  double value = 0;

  file.open(filename, std::ios::binary);
  if (!file.is_open()){
    std::cout << "Could not save Image in file: " << filename << "! Exit" << std::endl;
    std::exit(1);
  }

  file << "P5" << std::endl;
  file << size.first << " " << size.second << std::endl;
  file << "255" << std::endl;
  for (int j = 0; j < size.second; j++){
    for (int i = 0; i < size.first; i++){
      // clamp values if < 0 and > 255
      value = src.At(i,j, false);
      value = (value > 255) ? 255 : value;
      value = (value < 0) ? 0 : value;
      file << (unsigned char)value;
    }
  }
  file.close();
}


void writePNGImage(std::string filename, Img &src){
  std::pair<int, int> size = src.Size();
  std::vector<unsigned char> raw (size.first * size.second * 4);
  double value = 0;

  for (int j = 0; j < size.second; j++){
    for (int i = 0; i < size.first; i++){
      // clamp values if < 0 and > 255
      value = src.At(i,j, false);
      value = (value > 255) ? 255 : value;
      value = (value < 0) ? 0 : value;

      raw[j * size.first * 4 + i * 4] = (unsigned char) value;
      raw[j * size.first * 4 + i * 4 + 1] = (unsigned char) value;
      raw[j * size.first * 4 + i * 4 + 2] = (unsigned char) value;
      raw[j * size.first * 4 + i * 4 + 3] = (unsigned char) 255;
    }
  }

  lodepng::encode(filename.c_str(), raw, size.first, size.second);
}


void loadBarronFile(std::string filename, FlowField &dest){

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

  // resize dest
  dest.Resize(nx, ny);

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
  for (int j = 0; j < ny; j++){
    for (int i = 0; i < nx; i++){
      dest.u.Set(i, j, tmpu[i + offsetx][j + offsety]);
      dest.v.Set(i, j, tmpv[i + offsetx][j + offsety]);
    }
  }

}
