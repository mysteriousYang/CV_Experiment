#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace ml;

namespace rtree_space {
	Ptr<TrainData> __create_fDataset();
}

Ptr<TrainData> rtree_space::__create_fDataset() {
	Mat img = imread("olivettifaces.png", IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "Can't load olivettifaces.png" << std::endl;
		abort();
	}

	Mat data, label;

	for (int i = 0; i < 20; i++) {
		int offsetRow = i * 57; //���ϵ�ƫ����
		for (int j = 0; j < 20; j++) {
			int offsetCol = j * 47;  //���ϵ�ƫ����
			Mat tmp;
			img(Range(offsetRow, offsetRow + 57), Range(offsetCol, offsetCol + 47)).copyTo(tmp);

			data.push_back(tmp.reshape(0, 1));
			label.push_back((int)j / 10 + 2 * i);
			//label.at<float>(i * 20 + j, j / 10 + 2 * i) = 100.0;

			//reshape  0��ͨ������  �������֣���ʾҪ���õ�ͨ����
			//reshape  1: ��ʾ�����������������Ϊ0�����ʾ����ԭ���������䣬�������Ϊ�������֣���ʾҪ���õ�����
		}
	}
	data.convertTo(data, CV_32F);
	//label.convertTo(label, CV_32S);
	//std::cout << label << std::endl;
	return TrainData::create(data, ROW_SAMPLE, label);
}

void my_rtree() {
	using std::cout; using std::endl;
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	TimeCounter timer;

	//ע�����ɭ�ַ������ı�ǩ����������
	auto dataSet = rtree_space::__create_fDataset();
	double trRatio = 0.7;
	dataSet->setTrainTestSplitRatio(trRatio, true/*for shuffle*/);
	dataSet->shuffleTrainTest();

	// ����������
	auto model = RTrees::create();
	//�������������
	model->setMaxDepth(7);
	//�ڵ���С��������
	model->setMinSampleCount(10);
	//�ع�������ֹ��׼
	//model->setRegressionAccuracy(0);
	//�Ƿ���������ѵ�
	model->setUseSurrogates(false);
	//���������
	//model->setMaxCategories(100);
	//�������������
	model->setPriors(Mat());
	//����ı�����Ҫ��
	model->setCalculateVarImportance(true);
	//���ڵ����ѡ��������Ӽ��Ĵ�С
	//model->setActiveVarCount(100);
	//��ֹ��׼
	model->setTermCriteria(
		TermCriteria(TermCriteria::Type::COUNT, 100, 0.001)
	);
	
	//ѵ��ģ��
	cout << "Start training" << endl;
	timer.Start();
	model->train(dataSet);
	timer.Stop();
	cout << "Training cost : " << timer.RunTime << " ms" << endl;

	auto tr_data = dataSet->getTrainSamples();
	auto tr_lab = dataSet->getTrainResponses();
	auto ts_data = dataSet->getTestSamples();
	auto ts_lab = dataSet->getTestResponses();

	printf("Calculating Performance\n");
	double train_hr = 0, test_hr = 0;
	Point pos;
	//calc train
	for (int i = 0; i < tr_data.rows; ++i) {
		int pred = model->predict(tr_data.row(i), noArray());
		int label = tr_lab.at<int>(i);
		if (pred == label) {
			++train_hr;
		}
		else {
			//printf("Failed train[%3d] Pred: %2d, Label: %2d\n", i, pred, label);
		}
	}
	//calc test
	for (int i = 0; i < ts_data.rows; ++i) {
		int pred = model->predict(ts_data.row(i), noArray());
		int label = ts_lab.at<int>(i);
		if (pred == label) {
			++test_hr;
		}
		else {
			//printf("Failed test[%3d] Pred: %2d, Label: %2d\n", i, pred, label);
		}
	}
	printf("Train prec : %.1f%%\n", 100 * train_hr / tr_data.rows);
	printf("Test  prec : %.1f%%\n", 100 * test_hr / ts_data.rows);

	for (int i = 1;; ++i) {
		int imgIdx = rand() % tr_data.rows;
		Mat img = tr_data.row(imgIdx);
		int pd = model->predict(img);
		img.reshape(0, 57).convertTo(img, CV_8U);
		img_show("Trial " + std::to_string(i), img, 300, 360);
		printf("Trial %2d Pred: %2d, Lab: %2d\n", i, pd, tr_lab.at<int>(imgIdx));
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
	
}