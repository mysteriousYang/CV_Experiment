#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

int my_sift() {
	srand((unsigned)new char);
	std::default_random_engine rd_e;

	auto gauss_noise = [&rd_e](double _u = 0, double _s = 15) -> double {
		std::normal_distribution<double> n_noise(_u, _s);
		return n_noise(rd_e);
	};

	//Mat src_scaled, src_rotated, src_noised;
	//[0]:origin [1]:scaled [2]:rotated [3]:noised
	Mat imgs[4];
	imgs[0] = imread("eyjafjalla.jpg", IMREAD_COLOR);
	cvtColor(imgs[0], imgs[0], COLOR_BGR2GRAY);
	
	string _window_names[] = {
	"Gray Eyjafjalla","Scaled Eyjafjalla","Rotated Eyjafjalla","Noised Eyjafjalla"
	};

	
	//scale
	resize(imgs[0], imgs[1], {960,540});

	//rotation center
	double angle = 45;
	Point2f center((imgs[0].cols - 1) / 2.0, (imgs[0].rows - 1) / 2.0);
	Mat rotation_matix = getRotationMatrix2D(center, angle, 1.0);

	//rotate
	warpAffine(imgs[0], imgs[2], rotation_matix, imgs[0].size());

	//noise
	Mat noise(imgs[0].rows, imgs[0].cols, CV_8S);
	for (auto it = noise.begin<char>(); it != noise.end<char>(); ++it) {
		*it = (char)gauss_noise(0,40);
	}
	imgs[0].copyTo(imgs[3]);
	imgs[3] += noise;

	//show original images
	//for (int i = 0; i < 4; ++i) {
	//	img_show(_window_names[i], imgs[i], 720, 405);
	//}
	//waitKey(0);


	//do sift detect
	auto sift_detector = SIFT::create();
	std::vector<KeyPoint> kps[4];

	Mat descriptors[4];
	for (int i = 0; i < 4; ++i) {
		sift_detector->detectAndCompute(imgs[i], noArray(), kps[i], descriptors[i]);
	}

	Mat sifted_imgs[4];
	for (int i = 0; i < 4; ++i) {
		drawKeypoints(imgs[i], kps[i], sifted_imgs[i]);
	}


	//show sifted imgs
	for (int i = 0; i < 4; ++i) {
		img_show(_window_names[i], sifted_imgs[i], 720, 405);
	}

	waitKey(0);
	return 0;
}