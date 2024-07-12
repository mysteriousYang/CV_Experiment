#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

void my_pyrMeanShift() {
	Mat src = imread("_MG_7526.jpg", IMREAD_COLOR);
	//cvtColor(src, src, COLOR_BGR2GRAY);

	Mat meanShift_src, canny_src;
	pyrMeanShiftFiltering(src, meanShift_src, 15, 60);

	img_show("MeanShifted Maotiao", meanShift_src, 720, 480);
	//imwrite("meanshifted_maotiao.png", meanShift_src);

	Canny(meanShift_src, canny_src, 10, 10 * 3, 3);
	img_show("Canny Detector", canny_src, 720, 480);

	waitKey();
}