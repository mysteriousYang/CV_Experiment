#ifndef _MY_UTIL_
#define _MY_UTIL_
#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

void img_show(const string& _window_name, const Mat& _img, uint _ww, uint _wh) {
	namedWindow(_window_name, WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
	resizeWindow(_window_name, _ww, _wh);
	imshow(_window_name, _img);
	//waitKey(0);
}

#endif