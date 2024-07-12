#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

int thresh = 200;
int max_thresh = 255;

void cornerHarris_demo(const string& memo, Mat& src);

int my_harris() {
	srand((unsigned)new char);
	Mat src, src_gray, src_scaled, src_rotated, src_noised;
	double angle = 45; //deg

	std::default_random_engine rd_e;

	auto gauss_noise = [&rd_e](double _u = 0, double _s = 15) -> double {
		std::normal_distribution<double> n_noise(_u, _s);
		return n_noise(rd_e);
	};
	

	src = imread("eyjafjalla.jpg", IMREAD_COLOR);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	//scale
	resize(src_gray, src_scaled, { 960,540 });
	//img_show("Scaled Eyjafjalla", src_scaled);

	//rotation center
	Point2f center((src_gray.cols - 1) / 2.0, (src_gray.rows - 1) / 2.0);
	Mat rotation_matix = getRotationMatrix2D(center, angle, 1.0);

	//rotate
	warpAffine(src_gray, src_rotated, rotation_matix, src_gray.size());
	//img_show("Rotated Eyjafjalla", src_rotated);

	//noise
	Mat noise(src_gray.rows, src_gray.cols, CV_8S);
	for (auto it = noise.begin<char>(); it != noise.end<char>(); ++it) {
		*it = (char)gauss_noise();
	}
	src_gray.copyTo(src_noised);
	src_noised += noise;
	//img_show("Noised Eyjafjalla", src_noised);

	//namedWindow("Source Window");
	img_show("Source Eyjafjalla", src_gray, 720, 404);

	cornerHarris_demo("Scaled Eyjafjalla", src_scaled);
	cornerHarris_demo("Rotated Eyjafjalla", src_rotated);
	cornerHarris_demo("Noised Eyjafjalla", src_noised);

	waitKey(0);
	return 0;
}

void cornerHarris_demo(const string& memo, Mat& src){
	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	Mat dst = Mat::zeros(src.size(), CV_32FC1);
	cornerHarris(src, dst, blockSize, apertureSize, k);

	/// Normalizing
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int i = 0; i < dst_norm.rows; i++){
		for (int j = 0; j < dst_norm.cols; j++){
			if ((int)dst_norm.at<float>(i, j) > thresh){
				circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
			}
		}
	}

	/// Showing the result
	img_show(memo, dst_norm_scaled, 720, 404);
}