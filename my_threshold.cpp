#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

double _do_threshold(const Mat& _src,Mat& _dst,double _th) {
	double m1 = 0, m2 = 0;
	int m1_count = 0, m2_count = 0;
	_dst.create(_src.size(), IMREAD_GRAYSCALE);
	for (int i = 0; i < _src.rows; ++i) {
		for (int j = 0; j < _src.cols; ++j) {
			if (_src.at<uchar>(i, j) > _th) {
				_dst.at<uchar>(i, j) = 0xFF;
				++m1_count;
				m1 += _src.at<uchar>(i, j);
			}
			else {
				_dst.at<uchar>(i, j) = 0x00;
				++m2_count;
				m2 += _src.at<uchar>(i, j);
			}
		}
	}
	m1 /= m1_count;
	m2 /= m2_count;
	return (m1 + m2) / 2.0;
}

void my_threshold() {
	auto src = imread("ganyu.jpg", IMREAD_COLOR);
	Mat dst;
	cvtColor(src, src, COLOR_BGR2GRAY);

	img_show("ganyu",src);
	double dt = UINT32_MAX;
	double th_0 = 127, th_1 = 0;
	do {
		th_1 = _do_threshold(src, dst, th_0);
		dt = std::fabs(th_1 - th_0);
		th_0 = th_1;
		std::cout << "Now threshold = " << th_0 << std::endl;
	} while (dt >= 10);
	string window_title = "Threshold = " + std::to_string(th_0);
	img_show(window_title, dst);
	waitKey(0);
}