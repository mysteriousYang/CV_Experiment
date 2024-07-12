#ifndef _MY_UTIL_
#define _MY_UTIL_
#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

typedef Point_<uchar> Pixel;
std::default_random_engine rd_e;

void img_show(const string& _window_name, const Mat& _img, uint _ww, uint _wh) {
	namedWindow(_window_name, WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
	resizeWindow(_window_name, _ww, _wh);
	imshow(_window_name, _img);
	//waitKey(0);
}

inline double _gauss_noise(double _u, double _s){
	return std::normal_distribution<double>(_u, _s)(rd_e);
}

Mat gauss_image(size_t _rows, size_t _cols, double _u, double _s) {
	Mat gaussian_image(_rows, _cols, CV_8S);
	gaussian_image.forEach<char>([&](char& _pixel, const int* _position) -> void {
		_pixel = _gauss_noise(_u, _s);
		});
	return gaussian_image;
}


#endif