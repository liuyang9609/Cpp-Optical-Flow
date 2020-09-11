#include "image_class.h"


BaseArray::BaseArray(int sizex, int sizey){
  Resize(sizex, sizey);
}

BaseArray::BaseArray(){
}

double BaseArray::operator() (int x, int y, bool mirror){
  return At(x, y, mirror);
}

std::vector<double>& BaseArray::operator[] (int x){
  return data[x];
}

double BaseArray::At (int x, int y, bool mirror){
  std::pair<int, int> size = Size();

  // mirror on boundaries if flag is true
  if (mirror){
    int newx;
    int newy;

    // new xpos
    if (x < 0){
      newx = fabs(x);
    } else if (x > size.first - 1) {
      newx = 2 * size.first - x - 2;
    } else {
      newx = x;
    }

    // new ypos
    if (y < 0){
      newy = fabs(y);
    } else if (y > size.second - 1) {
      newy = 2 * size.second - y - 2;
    } else {
      newy = y;
    }

    return data[newx][newy];
  } else {

    if (x < 0 || x > size.first - 1 || y < 0 || y > size.second - 1){
      return 0;
    } else {
      return data[x][y];
    }
  }
}


void BaseArray::Set (int x, int y, double value){
  data[x][y] = value;
}


void BaseArray::SetAll (double value){
  std::pair<int, int> s = Size();

  for (int i = 0; i < s.first; i++) {
    for (int j = 0; j < s.second; j++){
      Set(i, j, value);
    }
  }
}

void BaseArray::SetTo (BaseArray *src){
  if (Size() != src->Size()){
    std::cout << "Image size in SetTo not equal. Exit" << std::endl;
    std::exit(1);
  }

  std::pair<int, int> size = Size();
  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      Set(i, j, (*src->getData())[i][j]);
    }
  }
}


void BaseArray::SetTo (std::vector< std::vector<double> > src){

  std::pair<int, int> size = Size();

  if (src.size() <= 0){
    std::cout << "Error in SetTo. Source vector empty" << std::endl;
    std::exit(1);
  }

  if (src.size() != size.first || src[0].size() != size.second){
    std::cout << "Image dimensions dont match in BaseArray::SetTo" << std::endl;
    std::exit(1);
  }

  // do not directly set vector to use probably extra stuff (guards etc) in Set
  for (int i = 0; i < src.size(); i++){
    for (int j = 0; j < src[0].size(); j++){
      Set(i, j, src[i][j]);
    }
  }
}

std::vector< std::vector<double> > *BaseArray::getData(){
  // return it as a pointer
  return &data;
}


void BaseArray::Resize (int x, int y){
  data.resize(x);
  for (auto &i: data){
    i.resize(y);
  }
}

void BaseArray::Resize (std::pair<int, int> size){
  data.resize(size.first);
  for (auto &i: data){
    i.resize(size.second);
  }
}


std::pair<int, int> BaseArray::Size(){
  std::pair<int, int> s;
  s.first = data.size();
  s.second = data[0].size();

  return s;
}


void BaseArray::ResampleLanczos(int sizex, int sizey, int n){
  double xratio, yratio, sum, weight, x, y, Lx, Ly;
  std::pair<int, int> size = Size();

  std::vector< std::vector<double> > tmp (sizex, std::vector<double>(sizey));
  xratio = sizex/(double)size.first;
  yratio = sizey/(double)size.second;

  for (int i = 0; i < sizex; i++){
    for (int j = 0; j < sizey; j++){
      x = i/xratio;
      y = j/yratio;
      sum = 0;
      weight = 0;

      for (int r = -n+1; r < n; r++){
        for (int s = -n+1; s < n; s++){
          int cx = floor(x)+r, cy = floor(y)+s;
          Lx = Lanczos(cx-x, n);
          Ly = Lanczos(cy-y, n);
          weight += Lx * Ly;
          sum += At(cx, cy, true) * Lx * Ly;
        }
      }
      sum = sum/weight;
      sum = (sum < 0) ? 0 : sum;
      sum = (sum > 255) ? 255 : sum;
      tmp[i][j] = sum;
    }
  }
  data = tmp;
}

void BaseArray::ResampleArea(int sizex, int sizey){
  double xratio, yratio;
  std::pair<int, int> size = Size();

  // make a temporary array
  std::vector< std::vector<double> > tmp;
  tmp.resize(sizex);
  for (auto &i: tmp){
    i.resize(sizey);
  }

  // get ratio of resampling in each direction
  xratio = sizex/(double)size.first;
  yratio = sizey/(double)size.second;

  for (int i = 0; i < sizex; i++){
    for (int j = 0; j < sizey; j++){
      double x = i/xratio, y = j/yratio;
      double xnext = (i+1)/xratio, ynext = (j+1)/yratio;
      double sum = 0, sumweight = 0;

      for (int cx = floor(x); cx <= floor(xnext); cx++){
        for (int cy = floor(y); cy <= floor(ynext); cy++){

          // calculate the area of the current pixel which is covered by the resampled pixel
          double weight = AreaCovered(cx, cy, x, xnext, y, ynext);
          sumweight += weight;
          sum += weight * At(cx, cy, true);
        }
      }
      tmp[i][j] = sum/sumweight;
    }
  }
  data = tmp;
}

/* bilinear upsampling (for later)
int xf = floor(xcoord),
    xc = ceil(xcoord),
    yf = floor(ycoord),
    yc = ceil(ycoord);
double x1 = At(xf, yf, true),
       x2 = At(xc, yf, true),
       x3 = At(xf, yc, true),
       x4 = At(xc, yc, true);

tmp[i][j] = BilinearInterpolation(x1, x2, x3, x4, xcoord - xf, ycoord - yf);
*/


double BaseArray::Lanczos(double x, int a){
  if (x == 0) { return 1; }
  if (fabs(x) > 0 && fabs(x) < a) {
    return (a * sin(M_PI * x) * sin((M_PI * x)/a))/(pow(M_PI, 2) * pow(x, 2));
  }
  return 0;
}


double BaseArray::Gaussian(double x, double sigma) {
  double i;
  i = 1.0/(sqrt(2 * M_PI) * sigma);
  i = i * exp(-pow(x,2)/(2*pow(sigma,2)));
  return i;
}


void BaseArray::GaussianSmooth(double sigma){

  std::vector< std::vector<double> > tmp;

  // make array for gaussian values
  int size = ceil(6 * sigma);
  if (size % 2 == 0) {
    size += 1;
  }
  double gauss[size];
  double weight = 0;
  for (int i = 0; i < size; i++){
    gauss[i] = Gaussian(i - (size - 1)/2, sigma);
    weight += gauss[i];
  }
  // normalize
  for (auto &i: gauss){
    i = i/weight;
  }

  // make sure the gauss filter is not bigger the the image itself
  std::pair<int, int> s = Size();
  if (s.first < size){
    std::cout << "x-direction to low" << std::endl;
    std::exit(1);
  }
  if (s.second < size){
    std::cout << "y-direction to low" << std::endl;
    std::exit(1);
  }

  tmp.resize(s.first);
  for (auto &i: tmp){
    i.resize(s.second);
  }

  // smoothing in x-Direction
  for (int i=0; i<s.first; i++){
    for (int j=0; j<s.second; j++){
      double sum = 0;
      for (int k=0; k<size; k++){
        sum += gauss[k] * At(i+k-(size-1)/2, j, true);
      }
      tmp[i][j] = sum;
    }
  }
  data = tmp;

  // smoothing in y-direction
  for (int i=0; i<s.first; i++){
    for (int j=0; j<s.second; j++){
      double sum = 0;
      for (int k=0; k<size; k++){
        sum += gauss[k] * At(i, j+k-(size-1)/2, true);
      }
      tmp[i][j] = sum;
    }
  }

  data = tmp;
}



void FlowField::Resize (int sizex, int sizey){
  u.Resize (sizex, sizey);
  v.Resize (sizex, sizey);
}

void FlowField::SetAll (double value){
  u.SetAll(value);
  v.SetAll(value);
}

void FlowField::Resize (std::pair<int, int> size){
  u.Resize(size);
  v.Resize(size);
}

void FlowField::ResampleArea (int sizex, int sizey){
  u.ResampleArea(sizex, sizey);
  v.ResampleArea(sizex, sizey);
}

void FlowField::ResampleLanczos (int sizex, int sizey, int n){
  u.ResampleLanczos(sizex, sizey, n);
  v.ResampleLanczos(sizex, sizey, n);
}

std::pair<int, int> FlowField::Size(){
  return u.Size();
}

std::vector< std::vector<RGBA> > FlowField::getColorImage(){

  // make temporary array
  std::pair<int, int> size = Size();
  std::vector< std::vector<RGBA> > img(size.first, std::vector<RGBA>(size.second));

  double Pi = M_PI;
  double amp;
  double phi;
  double alpha, beta;
  double x, y;

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      x = u.At(i, j, true);
      y = v.At(i, j, true);

      /* determine amplitude and phase (cut amp at 1) */
      amp = sqrt (x * x + y * y);
      if (amp > 1) amp = 1;
      if (x == 0.0)
        if (y >= 0.0) phi = 0.5 * Pi;
        else phi = 1.5 * Pi;
      else if (x > 0.0)
        if (y >= 0.0) phi = atan (y/x);
        else phi = 2.0 * Pi + atan (y/x);
      else phi = Pi + atan (y/x);
      phi = phi / 2.0;

      img[i][j].a = 255;

      // interpolation between red (0) and blue (0.25 * Pi)
      if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
        beta  = phi / (0.125 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha *   0.0 + beta * 255.0));
      }
      if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
        beta  = (phi-0.125 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha * 255.0 + beta *  64.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha *   0.0 + beta *  64.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
      if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
        beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha *  64.0 + beta *   0.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha *  64.0 + beta * 255.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
      }
      if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
        beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha *   0.0 + beta *   0.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
      }
      // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
      if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
        beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha * 0.0   + beta * 255.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha * 0.0   + beta * 0.0));
      }
      // interpolation between yellow (0.75 * Pi) and red (Pi)
      if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
        beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        img[i][j].r = (unsigned char)floor(amp * (alpha * 255.0 + beta * 255.0));
        img[i][j].g = (unsigned char)floor(amp * (alpha * 255.0 + beta *   0.0));
        img[i][j].b = (unsigned char)floor(amp * (alpha * 0.0   + beta *   0.0));
      }

      /* check RGB range */
      img[i][j].r = byte_range((int)img[i][j].r);
      img[i][j].g = byte_range((int)img[i][j].g);
      img[i][j].b = byte_range((int)img[i][j].b);
    }
  }


  return img;
}


void FlowField::writeToPNG(std::string filename){
  std::vector< std::vector<RGBA> > src = getColorImage();
  std::pair<int, int> size = u.Size();
  std::vector<unsigned char> raw (size.first * size.second * 4);

  for (int j = 0; j < size.second; j++){
    for (int i = 0; i < size.first; i++){
      raw[j * size.first * 4 + i * 4] = src[i][j].r;
      raw[j * size.first * 4 + i * 4 + 1] = src[i][j].g;
      raw[j * size.first * 4 + i * 4 + 2] = src[i][j].b;
      raw[j * size.first * 4 + i * 4 + 3] = src[i][j].a;
    }
  }

  lodepng::encode(filename.c_str(), raw, size.first, size.second);
}

void FlowField::writeErrorToPNG(std::string filename, FlowField &t){
  FlowField tmp;
  std::pair<int, int> size = Size();
  tmp.Resize(size);

  for (int i = 0; i < size.first; i++){
    for (int j = 0; j < size.second; j++){
      tmp.u[i][j] = u[i][j] - t.u[i][j];
      tmp.v[i][j] = v[i][j] - t.v[i][j];
    }
  }

  std::vector< std::vector<RGBA> > src = tmp.getColorImage();
  std::vector<unsigned char> raw (size.first * size.second * 4);

  for (int j = 0; j < size.second; j++){
    for (int i = 0; i < size.first; i++){
      raw[j * size.first * 4 + i * 4] = src[i][j].r;
      raw[j * size.first * 4 + i * 4 + 1] = src[i][j].g;
      raw[j * size.first * 4 + i * 4 + 2] = src[i][j].b;
      raw[j * size.first * 4 + i * 4 + 3] = src[i][j].a;
    }
  }

  lodepng::encode(filename.c_str(), raw, size.first, size.second);
}

double FlowField::CalcAngularError(FlowField &t){
  std::pair<int, int> size = Size();
  double amount = 0;
  double tmp1 = 0;
  double tmp2 = 0;
  double sum_l2 = 0;
  double sum_ang = 0;

  // make sure both flow field have same size
  if (size != t.Size()){
    std::cout << "Dimensions in CalcAngularError dont match" << std::endl;
    std::exit(1);
  }

  for (int i = 0; i < size.first; i++) {
    for (int j = 0; j < size.second; j++){
      // test if reference flow vector exists
      if (t.u[i][j] != 100.0 || t.v[i][j] != 100.0){
        amount++;
        tmp1 = t.u[i][j] - u[i][j];
        tmp2 = t.v[i][j] - v[i][j];

        sum_l2 += sqrt(tmp1*tmp1 + tmp2*tmp2);
        tmp1 = (t.u[i][j] * u[i][j] + t.v[i][j] * v[i][j] + 1.0)
   		     / sqrt( (t.u[i][j] * t.u[i][j] + t.v[i][j] * t.v[i][j] + 1.0)
   			     * (u[i][j] * u[i][j] + v[i][j] * v[i][j] + 1.0) );

        if (tmp1 > 1.0) tmp1 = 1.0;
        if (tmp1 < - 1.0) tmp1 = - 1.0;
        sum_ang += acos(tmp1) * 180.0/M_PI;
      }
    }
  }

  return sum_ang / amount;
}


double Img::AtOriginal(int x, int y, bool mirror){
  std::pair<int, int> size = SizeOriginal();

  // mirror on boundaries if flag is true
  if (mirror){
    int newx;
    int newy;

    // new xpos
    if (x < 0){
      newx = fabs(x);
    } else if (x > size.first - 1) {
      newx = 2 * size.first - x - 2;
    } else {
      newx = x;
    }

    // new ypos
    if (y < 0){
      newy = fabs(y);
    } else if (y > size.second - 1) {
      newy = 2 * size.second - y - 2;
    } else {
      newy = y;
    }

    return original[newx][newy];
  } else {

    if (x < 0 || x > size.first - 1 || y < 0 || y > size.second - 1){
      return 0;
    } else {
      return original[x][y];
    }
  }
}

void Img::SetOriginal(int x, int y, double value){
  original[x][y] = value;
}

void Img::SetOriginalTo(std::vector< std::vector<double> > src){
  original = src;
}

void Img::ResizeOriginal(int sizex, int sizey){
  original.resize(sizex);
  for (auto &i: original){
    i.resize(sizey);
  }
}


void Img::ResampleArea(int sizex, int sizey){
  // override standart implementation, because we always resample from original image

  double xratio, yratio;
  std::pair<int, int> size = SizeOriginal();

  // resize data array
  Resize(sizex, sizey);


  // get ratio of resampling in each direction
  xratio = sizex/(double)size.first;
  yratio = sizey/(double)size.second;

  for (int i = 0; i < sizex; i++){
    for (int j = 0; j < sizey; j++){
      double x = i/xratio, y = j/yratio;
      double xnext = (i+1)/xratio, ynext = (j+1)/yratio;
      double sum = 0, sumweight = 0;

      for (int cx = floor(x); cx <= floor(xnext); cx++){
        for (int cy = floor(y); cy <= floor(ynext); cy++){

          // calculate the area of the current pixel which is covered by the resampled pixel
          double weight = AreaCovered(cx, cy, x, xnext, y, ynext);
          sumweight += weight;
          sum += weight * AtOriginal(cx, cy, true);

        }
      }
      Set(i, j, sum/sumweight);
    }
  }
}

void Img::ResampleLanczos(int sizex, int sizey, int n){
  double xratio, yratio, sum, weight, x, y, Lx, Ly;
  std::pair<int, int> size = SizeOriginal();

  std::vector< std::vector<double> > tmp (sizex, std::vector<double>(sizey));
  xratio = sizex/(double)size.first;
  yratio = sizey/(double)size.second;

  for (int i = 0; i < sizex; i++){
    for (int j = 0; j < sizey; j++){
      x = i/xratio;
      y = j/yratio;
      sum = 0;
      weight = 0;

      for (int r = -n+1; r < n; r++){
        for (int s = -n+1; s < n; s++){
          int cx = floor(x)+r;
          int cy = floor(y)+s;
          Lx = Lanczos(cx-x, n);
          Ly = Lanczos(cy-y, n);
          weight += (Lx * Ly);
          sum += (AtOriginal(cx, cy, true) * Lx * Ly);
        }
      }
      sum = sum/weight;
      /*if (sum < 0){
        x = floor(x);
        y = floor(y);
        sum = AtOriginal(x, y, true);
        //sum = 0.25 * (AtOriginal(x-1, y-1, true) + AtOriginal(x-1, y+1, true) + AtOriginal(x+1, y-1, true) + AtOriginal(x+1, y+1, true));
      } else if (sum > 255) {
        x = floor(x);
        y = floor(y);
        sum = AtOriginal(x, y, true);
        //sum = 0.25 * (AtOriginal(x-1, y-1, true) + AtOriginal(x-1, y+1, true) + AtOriginal(x+1, y-1, true) + AtOriginal(x+1, y+1, true));
      }
      */
      sum = (sum < 0) ? 0 : sum;
      sum = (sum > 255) ? 255 : sum;
      tmp[i][j] = sum;
    }
  }
  data = tmp;
}


std::pair<int, int> Img::SizeOriginal(){
  std::pair<int, int> s;
  s.first = original.size();
  s.second = original[0].size();
  return s;
}

std::vector< std::vector<double> > Img::getOriginal(){
  return original;
}

double Img::Hx(){
  return SizeOriginal().first/(double)Size().first;
}

double Img::Hy(){
  return SizeOriginal().second/(double)Size().second;
}

void Img::GaussianSmoothOriginal(double sigma) {
  std::vector< std::vector<double> > tmp;

  // make array for gaussian values
  int size = ceil(6 * sigma);
  if (size % 2 == 0) {
    size += 1;
  }
  double gauss[size];
  double weight = 0;
  for (int i = 0; i < size; i++){
    gauss[i] = Gaussian(i - (size - 1)/2, sigma);
    weight += gauss[i];
  }
  // normalize
  for (auto &i: gauss){
    i = i/weight;
  }

  // make sure the gauss filter is not bigger the the image itself
  std::pair<int, int> s = SizeOriginal();
  if (s.first < size){
    std::cout << "x-direction to low" << std::endl;
    std::exit(1);
  }
  if (s.second < size){
    std::cout << "y-direction to low" << std::endl;
    std::exit(1);
  }

  tmp.resize(s.first);
  for (auto &i: tmp){
    i.resize(s.second);
  }

  // smoothing in x-Direction
  for (int i=0; i<s.first; i++){
    for (int j=0; j<s.second; j++){
      double sum = 0;
      for (int k=0; k<size; k++){
        sum += gauss[k] * AtOriginal(i+k-(size-1)/2, j, true);
      }
      tmp[i][j] = sum;
    }
  }
  original = tmp;

  // smoothing in y-direction
  for (int i=0; i<s.first; i++){
    for (int j=0; j<s.second; j++){
      double sum = 0;
      for (int k=0; k<size; k++){
        sum += gauss[k] * AtOriginal(i, j+k-(size-1)/2, true);
      }
      tmp[i][j] = sum;
    }
  }

  original = tmp;
}
