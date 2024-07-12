#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

void img_open() {
	srand((unsigned)new char);
	Mat img = imread("eyjafjalla.jpg", IMREAD_COLOR);

	std::default_random_engine rd_e;
	std::normal_distribution<double> n_noise(0, 15);

	img_show("Eyjafjalla", img);
	
	Mat gray_img;
	cvtColor(img, gray_img, COLOR_BGR2GRAY);
	img_show("Gray Eyjafjalla", gray_img);

	Mat noise(img.rows, img.cols, CV_8S);
	for (auto it = noise.begin<char>(); it != noise.end<char>(); ++it) {
		*it = (char)(n_noise(rd_e));
		//printf("%d\n", *it);
	}
	img_show("Noise", noise);
	imwrite("gaussian_noise.png", noise);
	
	gray_img += noise;
	img_show("Noised Ejyafjalla", gray_img);
	imwrite("eyjafjalla_noised.png", gray_img);

	waitKey(0);
	return;
}