#ifndef _MY_UTIL_HPP_
#define _MY_UTIL_HPP_
#pragma comment(lib, "opencv_world460d.lib")
#include<iostream>
#include<random>
#include<algorithm>
#include<string>
#include<vector>
#include<opencv2/features2d.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;
using std::string;



void img_show(const string& _window_name, const Mat& _img, uint _ww = 763, uint _wh = 540);
inline double _gauss_noise(double _u = 0, double _s = 15);
Mat gauss_image(size_t _rows, size_t _cols, double _u = 0, double _s = 15);

#endif // !_MY_UTIL_HPP_
