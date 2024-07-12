#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace ml;

void my_svm() {
	using std::cout; using std::endl;
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	Mat trainData, trainLabels;
	Mat testData, testLabels;
	load_Olivetti_Face(trainData, trainLabels, testData, testLabels, CV_32S);


	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	auto model = SVM::create();

	model->setType(SVM::C_SVC);
	model->setKernel(SVM::KernelTypes::LINEAR);
	//model->setDegree(1);
	//model->setGamma(1);
	//model->setCoef0(0);
	//model->setC(1);
	//model->setNu(0);
	//model->setP(0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 0.01));

	model->train(tData);

	//Ԥ�����
	double train_hr = 0, test_hr = 0;
	float r;
	Mat response;
	//ѵ����
	for (int i = 0; i < trainData.rows; ++i)
	{
		Mat sample = trainData.row(i);
		r = model->predict(sample);   //�������н���Ԥ��
									  //Ԥ�������ǩ��ȣ����Ϊ1������Ϊ0
		r = std::abs(r - trainLabels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		train_hr += r;  //�ۻ���ȷ��
	}

	//���Լ�
	for (int i = 0; i < testData.rows; ++i) {
		Mat sample = testData.row(i);
		r = model->predict(sample);
		r = std::abs(r - testLabels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		test_hr += r;
	}

	test_hr /= testData.rows;
	train_hr /= trainData.rows;

	printf("accuracy: train = %.1f%%, test = %.1f%%\n",
		train_hr * 100., test_hr * 100.);
}