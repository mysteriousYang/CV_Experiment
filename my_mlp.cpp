#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace ml;

namespace mlp_space {
	string __layer2str(const Mat& _layerMat);
	Ptr<TrainData> __create_fDataset();
}

string mlp_space::__layer2str(const Mat& _layerMat) {
	string str;
	for (auto i = _layerMat.begin<int>(); i != _layerMat.end<int>(); ++i) {
		str += std::to_string(*i);
		str += '-';
	}
	str.pop_back();
	return str;
}


Ptr<TrainData> mlp_space::__create_fDataset() {
	Mat img = imread("olivettifaces.png", IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "Can't load olivettifaces.png" << std::endl;
		abort();
	}

	//由于cv::MLP不能显式支持分类任务，所以只能设置40个桶
	Mat data, label(Mat::zeros(20 * 20, 40, CV_32F));

	for (int i = 0; i < 20; i++) {
		int offsetRow = i * 57; //列上的偏移量
		for (int j = 0; j < 20; j++) {
			int offsetCol = j * 47;  //行上的偏移量
			Mat tmp;
			img(Range(offsetRow, offsetRow + 57), Range(offsetCol, offsetCol + 47)).copyTo(tmp);
			
			data.push_back(tmp.reshape(0, 1));
			label.at<float>(i * 20 + j, j / 10 + 2 * i) = 1.0;

			//reshape  0：通道不变  其他数字，表示要设置的通道数
			//reshape  1: 表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
		}
	}
	data.convertTo(data, CV_32F);
	//label.convertTo(label, CV_32F);
	//std::cout << label << std::endl;
	return TrainData::create(data, ROW_SAMPLE, label);
}

void my_mlp() {
	using std::cout; using std::endl;
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	TimeCounter timer;

	auto dataSet = mlp_space::__create_fDataset();
	double trRatio = 0.8;
	dataSet->setTrainTestSplitRatio(trRatio, true/*for shuffle*/);
	dataSet->shuffleTrainTest();

	auto model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 4) << 47 * 57, 120, 80, 40);
	cout << "Layer: " << mlp_space::__layer2str(layerSizes) << endl;
	cout << "ActivationFunction: SIGMOID" << endl;
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	//model->setActivationFunction(ANN_MLP::GAUSSIAN, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 1000, 0.01));

	printf("Start Training\n");
	timer.Start();
	model->train(dataSet);
	timer.Stop();
	printf("Train Finished\n");
	cout << "Costed time: " << timer.RunTime << " ms" << endl << endl;

	auto tr_data = dataSet->getTrainSamples();
	auto tr_lab = dataSet->getTrainResponses();
	auto ts_data = dataSet->getTestSamples();
	auto ts_lab = dataSet->getTestResponses();

	//cout << tr_lab << endl;

	printf("Calculating Performance\n");
	double train_hr = 0, test_hr = 0;
	Point pos;
	//calc train
	for (int i = 0; i < tr_data.rows; ++i) {
		int pred = model->predict(tr_data.row(i), noArray());
		minMaxLoc(tr_lab.row(i), NULL, NULL, NULL, &pos);
		int label = pos.x;
		//int label = (i / (20 * trRatio)) * 2 + (i % int(20 * trRatio)) / 10;
		if (pred == label) {
			++train_hr;
		}
		else {
			//printf("Failed train[%d] Pred: %d, Label: %d\n", i, pred, label);
		}
	}
	//calc test
	for (int i = 0; i < ts_data.rows; ++i) {
		int pred = model->predict(ts_data.row(i), noArray());
		minMaxLoc(ts_lab.row(i), NULL, NULL, NULL, &pos);
		int label = pos.x;
		//int label = (i / (20 * (1 - trRatio))) * 2 + (i % int(20 * (1 - trRatio))) / 10;
		if (pred == label) {
			++test_hr;
		}
		else {
			//printf("Failed test[%d] Pred: %d, Label: %d\n", i, pred, label);
		}
	}
	printf("Train prec : %.3f\n", train_hr / tr_data.rows);
	printf("Test  prec : %.3f\n", test_hr / ts_data.rows);

	Mat response;
	for (int i = 1;; ++i) {
		int imgIdx = rand() % tr_data.rows;
		auto img = tr_data.row(imgIdx);
		int pred = model->predict(img, response);
		minMaxLoc(tr_lab.row(imgIdx), NULL, NULL, NULL, &pos);
		int label = pos.x;
		//int label = (imgIdx / 20) * 2 + (imgIdx % 20) / 10;

		img.reshape(0, 57).convertTo(img, CV_8U);
		img_show("Trial " + std::to_string(i), img, 300, 360);
		printf("Trial %d Pred: %d, Lab: %d\n", i, pred, label);

		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}