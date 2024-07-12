#pragma comment(lib, "opencv_world460d.lib")
#include<numeric>
#include<iterator>
#include"utilities.hpp"

using namespace ml;

class NormalDistribution {
private:
	float _u, _s, _s2;
	const float C1 = 0.3989422;
public:
	NormalDistribution(float u = 0, float s = 1)
		:_u(u), _s(s), _s2(s*s)
	{}
	float calc(float z) const{
		return C1 / _s * exp(-1.0 * (z - _u) * (z - _u) / (2 * _s2));
	}
	NormalDistribution& operator=(const NormalDistribution& _r) {
		_u = _r._u;
		_s = _r._s;
		_s2 = _r._s2;
		return *this;
	}
	float getU() const { return _u; }
	float getS()const { return _s; }
};

using N_d = NormalDistribution;

class NaiveBayes {
private:
	int _debug, _isTrained;
	TimeCounter _timer;
	int _classCounts, _features, _sampCounts;
	std::vector<int> _classSamp;//该训练集中某个类样本的数量
	std::vector<std::vector<N_d>> _norm/*每个类对应的正态分布列表*/;
	std::vector<std::vector<float>> _theta/*均值*/, _sigma/*方差*/;
	Ptr<TrainData> _tData;

	void _calcTheta();
	void _calcSigma();
	void _makeND();

public:
	NaiveBayes(int debug = 0)
		:_debug(debug) , _isTrained(0)
	{}

	static Ptr<TrainData> __create_fDataset();
	void		train(Ptr<TrainData> _dataSet);
	uint32_t    predict(const Mat& _sample) const;
	float		calcError(bool _test = false);
	void		calcErrorAll() {
		calcError(false);
		calcError(true);
	}
	double		getRuntime() { return _timer.RunTime; }
};

void NaiveBayes::_calcTheta() {
	auto X = _tData->getTrainSamples();
	auto Y = _tData->getTrainResponses();

	auto X_p = (float*)X.data;
	auto Y_p = (int*)Y.data;

	//calc theta
	for (int i = 0; i < _sampCounts; ++i) {
		uint32_t classIdx = Y_p[i];
		_classSamp[classIdx] += 1;
		float* row_p = X.ptr<float>(i);//获得第i行的行指针
		for (int j = 0; j < _features; ++j) {
			//printf("%f ", row_p[j]);
			_theta[classIdx][j] += row_p[j];
		}
		//putchar('\n');
	}
	for (int i = 0; i < _classCounts; ++i) {
		for (int j = 0; j < _features; ++j) {
			_theta[i][j] /= _classSamp[i];
		}
	}
}

void NaiveBayes::_calcSigma() {
	auto X = _tData->getTrainSamples();
	auto Y = _tData->getTrainResponses();

	auto X_p = (float*)X.data;
	auto Y_p = (int*)Y.data;

	//calc sigma
	for (int i = 0; i < _sampCounts; ++i) {
		uint32_t classIdx = Y_p[i];
		float* row_p = X.ptr<float>(i);//获得第i行的行指针
		for (int j = 0; j < _features; ++j) {
			//printf("%f ", row_p[j]);
			_sigma[classIdx][j] += (row_p[j] - _theta[classIdx][j]) * (row_p[j] - _theta[classIdx][j]);
		}
		//putchar('\n');
	}
	for (int i = 0; i < _classCounts; ++i) {
		for (int j = 0; j < _features; ++j) {
			_sigma[i][j] /= _classSamp[i];
			_sigma[i][j] = sqrt(_sigma[i][j]);
		}
	}

}

void NaiveBayes::_makeND() {
	//make NormalDistribution
	for (int i = 0; i < _classCounts; ++i) {
		for (int j = 0; j < _features; ++j) {
			//printf("[%d][%d] %6f %6f\n", i, j, _theta[i][j], _sigma[i][j]);
			_norm[i][j] = N_d(_theta[i][j], _sigma[i][j]);
		}
	}
}

void NaiveBayes::train(Ptr<TrainData> _dataSet) {
	//using std::cout; using std::endl;
	using std::vector;
	_tData = _dataSet;
	auto X = _tData->getTrainSamples();
	auto Y = _tData->getTrainResponses();

	if (Y.type() != CV_32S) {
		printf("Label must be int\n");
		exit(1);
	}
	if (X.rows != Y.rows) {
		printf("One sample must match one label\n");
		exit(2);
	}
	//if (!X.isContinuous()) {
	//	X = X.clone();
	//}

	_classCounts = 40;
	_features = X.cols;
	_sampCounts = X.rows;

	_theta.resize(_classCounts);
	_sigma.resize(_classCounts);
	_norm.resize(_classCounts);

	for (auto& row : _theta) {
		row.resize(_features, 0);
	}
	
	for (auto& row : _sigma) {
		row.resize(_features, FLT_EPSILON);//防止方差为0
	}

	for (auto& row : _norm) {
		row.resize(_features);
	}

	if (!_isTrained) {
		_classSamp.resize(_classCounts, 0);//该训练集中某个类样本的数量
	}



	if (_debug) {
		std::cout << "Train started" << std::endl;
	}
	_timer.Start();
	_calcTheta();
	_calcSigma();
	_makeND();
	_timer.Stop();

	_isTrained = 1;

	//output
	if (_debug) {
		for (auto& row : _sigma) {
			putchar('[ ');
			std::copy(row.begin(), row.end(), std::ostream_iterator<float>(std::cout, " "));
			putchar(' ]');
			putchar('\n');
		}
	}
	std::cout << "Training cost : " << _timer.RunTime << " ms" << std::endl;
	
}

uint32_t NaiveBayes::predict(const Mat& sample) const {
	using std::vector;
	vector<int> response;
	response.resize(_classCounts, 0);
	auto samp_ptr = sample.ptr<float>(0);

	for (int i = 0; i < _classCounts; ++i) {
		for (int j = 0; j < _features; ++j) {
			/// <judge>
			/// 由于直接计算P值乘积会造成类似“梯度消失”的问题
			/// 所以此处选择将测试数据标准化后，检查其在正态分布
			/// 中的位置。
			/// |u ± 2s|中包含了95%的可能
			/// |u ± 3s|中包含了99%的可能
			/// </judge>
			if (samp_ptr[j] - _norm[i][j].getU() < 3 * _norm[i][j].getS()) {
				if (_debug) {
					printf("Matched feature %4d\n", j);
				}
				response[i]++;
			}
		}
		continue;
	}
	if (_debug) {
		std::copy(response.begin(), response.end(), std::ostream_iterator<int>(std::cout, " "));
		std::cout << std::endl;
	}
	return std::distance(response.begin(), std::max_element(response.begin(), response.end()));
}

float NaiveBayes::calcError(bool _test) {
	Mat X, Y_true, Y_pred;
	float hr = 0;

	if (_test) {
		X = _tData->getTestSamples();
		Y_true = _tData->getTestResponses();
	}
	else {
		X = _tData->getTrainSamples();
		Y_true = _tData->getTrainResponses();
	}


	for (int i = 0; i < X.rows; ++i) {
		int pred = predict(X.row(i));
		int label = Y_true.at<int>(i);
		if (pred == label) {
			++hr;
		}
		else {
			if (_debug) {
				printf("Failed [%3d] Pred: %2d, Label: %2d\n", i, pred, label);
			}
		}
	}

	if (_test) {
		printf("Test  prec : %.1f%%\n", 100 * hr / X.rows);
	}
	else {
		printf("Train prec : %.1f%%\n", 100 * hr / X.rows);
	}

	return hr;
}

Ptr<TrainData> NaiveBayes::__create_fDataset() {
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

			//reshape  0：通道不变  其他数字，表示要设置的通道数
			//reshape  1: 表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
		}
	}
	data.convertTo(data, CV_32F);
	//label.convertTo(label, CV_32S);
	//std::cout << label << std::endl;
	return TrainData::create(data, ROW_SAMPLE, label);
}

void my_naiveBayes() {
	using std::cout; using std::endl;
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	auto dataSet = NaiveBayes::__create_fDataset();
	double trRatio = 0.8;
	dataSet->setTrainTestSplitRatio(trRatio, true/*for shuffle*/);
	dataSet->shuffleTrainTest();

	NaiveBayes model;
	model.train(dataSet);

	model.calcErrorAll();

	auto tr_data = dataSet->getTrainSamples();
	auto tr_lab = dataSet->getTrainResponses();
	auto ts_data = dataSet->getTestSamples();
	auto ts_lab = dataSet->getTestResponses();

	for (int i = 1;; ++i) {
		int imgIdx = rand() % ts_data.rows;
		Mat img = ts_data.row(imgIdx);
		int pd = model.predict(img);
		img.reshape(0, 57).convertTo(img, CV_8U);
		img_show("Trial " + std::to_string(i), img, 300, 360);
		printf("Trial %2d Pred: %2d, Lab: %2d\n", i, pd, ts_lab.at<int>(imgIdx));
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}

