#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

int nbayes_demo()
{
	// 样本数据
	int labels[4] = { 1, 1, -3, -1 };
	float trainingData[4][2] = { { 10, 10 },{ 10, 20 },{ 450, 100 },{ 150, 400 } };

	//	封装数据
	Mat labelsMat(4, 1, CV_32SC1, labels);
	Mat trainingDataMat(4, 2, CV_32F, trainingData);



	//	训练
	//	创建贝叶斯分类器
	Ptr<NormalBayesClassifier> Bayes = NormalBayesClassifier::create();
#if 1
	Bayes->train(trainingDataMat, ROW_SAMPLE, labelsMat);
#else
	Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	Bayes->train(tData);
#endif


	//	预测
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Show the decision regions given by the SVM
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(40, 80, 170);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = Bayes->predict(sampleMat);
			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
			else
				image.at<Vec3b>(i, j) = red;
		}
	}

	// Show the training data
	int thickness = -1;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness);

	imwrite("bayes-result.png", image);        // save the image
	imshow("Bayes Simple Example", image); // show it to the user

	waitKey();
	return 0;
}