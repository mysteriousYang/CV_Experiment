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
		int offsetRow = i * 57; //列上的偏移量
		for (int j = 0; j < 20; j++) {
			int offsetCol = j * 47;  //行上的偏移量
			Mat tmp;
			img(Range(offsetRow, offsetRow + 57), Range(offsetCol, offsetCol + 47)).copyTo(tmp);

			data.push_back(tmp.reshape(0, 1));
			label.push_back((int)j / 10 + 2 * i);
			//label.at<float>(i * 20 + j, j / 10 + 2 * i) = 100.0;

			//reshape  0：通道不变  其他数字，表示要设置的通道数
			//reshape  1: 表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
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

	//注意随机森林分类器的标签必须是整数
	auto dataSet = rtree_space::__create_fDataset();
	double trRatio = 0.7;
	dataSet->setTrainTestSplitRatio(trRatio, true/*for shuffle*/);
	dataSet->shuffleTrainTest();

	// 创建分类器
	auto model = RTrees::create();
	//树的最大可能深度
	model->setMaxDepth(7);
	//节点最小样本数量
	model->setMinSampleCount(10);
	//回归树的终止标准
	//model->setRegressionAccuracy(0);
	//是否建立替代分裂点
	model->setUseSurrogates(false);
	//最大聚类簇数
	//model->setMaxCategories(100);
	//先验类概率数组
	model->setPriors(Mat());
	//计算的变量重要性
	model->setCalculateVarImportance(true);
	//树节点随机选择的特征子集的大小
	//model->setActiveVarCount(100);
	//终止标准
	model->setTermCriteria(
		TermCriteria(TermCriteria::Type::COUNT, 100, 0.001)
	);
	
	//训练模型
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