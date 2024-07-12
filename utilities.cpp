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

void load_Olivetti_Face(
	Mat& _trainData, Mat& _trainLabels,
	Mat& _testData, Mat& _testLabels,
	int _LABEL_TY/*有些学习器要求标签矩阵的类型是Int，需要手动指定*/
){
	if (_trainData.empty() + _trainLabels.empty() + _testData.empty() + _testLabels.empty() != 4) {
		std::cout << "Input must be empty" << std::endl;
		abort();
	}

	Mat img = imread("olivettifaces.png", IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "Can't load olivettifaces.png" << std::endl;
		abort();
	}

	int w = 47, h = 57;
	int r_len = img.rows / h;
	int c_len = img.cols / w;

	int trainPerPerson = 8;

	for (int i = 0; i < r_len; i++) {
		int offsetRow = i * h; //列上的偏移量
		for (int j = 0; j < c_len; j++) {
			int offsetCol = j * w;  //行上的偏移量
			Mat tmp;
			img(Range(offsetRow, offsetRow + h), Range(offsetCol, offsetCol + w)).copyTo(tmp);
			if (
				(j < 10 && j < trainPerPerson)
				|| (j >= 10 && j - 10 < trainPerPerson)
				) {
				//push to train set
				_trainData.push_back(tmp.reshape(0, 1));
				_trainLabels.push_back((int)j / 10 + 2 * i);
			}
			else {
				//push to test set
				_testData.push_back(tmp.reshape(0, 1));
				_testLabels.push_back((int)j / 10 + 2 * i);
			}
			//reshape  0：通道不变  其他数字，表示要设置的通道数
			//reshape  1: 表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
		}
	}
	_trainData.convertTo(_trainData, CV_32F);
	_testData.convertTo(_testData, CV_32F);

	_trainLabels.convertTo(_trainLabels, _LABEL_TY);
	_testLabels.convertTo(_testLabels, _LABEL_TY);
}


#endif