#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"
//#include"TimerCounter.hpp"

const int MAX_CLUSTERS = 10;
Vec3b colorTab[] ={
	// Color in Roselia
	Vec3b(0x00, 0x22, 0xDD),
	Vec3b(0x88, 0x11, 0x88),
	Vec3b(0xBB, 0xAA, 0x00),
	Vec3b(0xBB, 0xBB, 0xBB),
	Vec3b(0x88, 0x00, 0xDD),

	// Color in Poppin Party
	Vec3b(0x22, 0x55, 0xFF),
	Vec3b(0xDD, 0x77, 0x00),
	Vec3b(0xBB, 0x55, 0xFF),
	Vec3b(0x11, 0xCC, 0xFF),
	Vec3b(0xDD, 0x66, 0xAA),

	Vec3b(0xFF, 0xCC, 0x66),
	Vec3b(0xCC, 0xFF, 0x66)};
RNG rng((unsigned) new char);

void my_kmeans() {
	for (;;) {
		Mat src = imread("ganyu.jpg", IMREAD_COLOR), labels, src_gray;
		cvtColor(src, src_gray, COLOR_BGR2GRAY);
		//TimerCounter timer;

		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		//int clusterCount = 5;
		int sampleCount = src.cols * src.rows;
		Mat points = src.reshape(1, sampleCount);
		points.convertTo(points, CV_32F);

		std::cout << "Processing K-Means" << std::endl;
		double compactness = kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS);

		std::cout << "Showing Image" << std::endl;
		Mat show(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				show.at<Vec3b>(i, j) = colorTab[labels.at<int>(i * src.cols + j)];
			}
		}

		img_show("K-Means Cluster = " + std::to_string(clusterCount), show, 508, 360);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}