#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

void my_otsu() {
	auto src = imread("ganyu.jpg", IMREAD_COLOR);
	cvtColor(src, src, COLOR_BGR2GRAY);
	Mat noised_src, blured_src;
	src.copyTo(noised_src);
	noised_src += gauss_image(noised_src.rows, noised_src.cols);
	GaussianBlur(noised_src, blured_src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Mat ostued[2];
	auto t1 = threshold(noised_src, ostued[0], 127, 255, THRESH_OTSU);
	auto t2 = threshold(blured_src, ostued[1], 127, 255, THRESH_OTSU);
	string window_names[] = {
		"Noised (Threshold = " + std::to_string(t1) + ")",
		"Blured (Threshold = " + std::to_string(t2) + ")",
		"Origin Ganyu"
	};
	img_show(window_names[2], src);
	img_show(window_names[0], ostued[0]);
	img_show(window_names[1], ostued[1]);
	waitKey();
}