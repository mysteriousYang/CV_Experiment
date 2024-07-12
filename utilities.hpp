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
#include<opencv2/ml.hpp>
#include<opencv2/core/utils/logger.hpp>

using namespace cv;
using std::string;

class TimeCounter {
	double _start;
public:
	double RunTime;
	void Start() {
		_start = getTickCount();
	}
	void Stop() {
		RunTime = (getTickCount() - _start) * 1000 / getTickFrequency();
	}
};

void img_show(const string& _window_name, const Mat& _img, uint _ww = 763, uint _wh = 540);
inline double _gauss_noise(double _u = 0, double _s = 15);
Mat gauss_image(size_t _rows, size_t _cols, double _u = 0, double _s = 15);
void load_Olivetti_Face(
	Mat& _trainData, Mat& _trainLabels,
	Mat& _testData, Mat& _testLabels,
	int _LABEL_TY = CV_32F
);

#endif // !_MY_UTIL_HPP_
