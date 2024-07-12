#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace ml;

void my_knn_digit() {
	Mat img = imread("digits.png", IMREAD_GRAYSCALE);
	int b = 20;
	int m = img.rows / b;   //原图为1000*2000
	int n = img.cols / b;   //裁剪为5000个20*20的小图块
	Mat data, labels;   //特征矩阵

	for (int i = 0; i < n; i++){
		int offsetCol = i * b; //列上的偏移量
		for (int j = 0; j < m; j++){
			int offsetRow = j * b;  //行上的偏移量
								  //截取20*20的小块
			Mat tmp;
			img(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			//reshape  0：通道不变  其他数字，表示要设置的通道数
			//reshape  表示矩阵行数，如果设置为0，则表示保持原有行数不变，如果设置为其他数字，表示要设置的行数
			data.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
			labels.push_back((int)j / 5);  //对应的标注
		}
	}

	data.convertTo(data, CV_32F); //uchar型转换为cv_32f
	int samplesNum = data.rows;
	int trainNum = 500;
	Mat trainData, trainLabels;
	trainData = data(Range(0, trainNum), Range::all());   //前3000个样本为训练数据
	trainLabels = labels(Range(0, trainNum), Range::all());

	//使用KNN算法
	int K = 5;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	auto model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
	//预测分类
	double train_hr = 0, test_hr = 0;
	Mat response;
	//计算误差
	for (int i = 0; i < samplesNum; i++)
	{
		Mat sample = data.row(i);
		float r = model->predict(sample);   //对所有行进行预测
											//预测结果与原结果相比，相等为1，不等为0
		r = std::abs(r - labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < trainNum)
			train_hr += r;  //累积正确数
		else
			test_hr += r;
	}

	test_hr /= samplesNum - trainNum;
	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

	printf("accuracy: train = %.1f%%, test = %.1f%%\n",
		train_hr * 100., test_hr * 100.);

	waitKey();
	for (int i = 1;;++i) {
		Mat randShowImg = data.row(rand() % samplesNum).reshape(0, 20);
		float pd = model->predict(randShowImg.reshape(0, 1));
		img_show("Trial " + std::to_string(i) + " Predicted as " + std::to_string((int)pd), randShowImg, 300, 300);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}

void my_knn_face() {
	//关闭OpenCV控制台输出
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	Mat trainData, trainLabels;
	Mat testData, testLabels;
	load_Olivetti_Face(trainData, trainLabels, testData, testLabels);

	//使用KNN算法
	int K = 5;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	auto model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
	//预测分类
	double train_hr = 0, test_hr = 0;
	float r;
	Mat response;
	//训练集
	for (int i = 0; i < trainData.rows; ++i)
	{
		Mat sample = trainData.row(i);
		r = model->predict(sample);   //对所有行进行预测
									  //预测结果与标签相比，相等为1，不等为0
		r = std::abs(r - trainLabels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		train_hr += r;  //累积正确数
	}

	//测试集
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

	for (int i = 1;; ++i) {
		int imgIdx = rand() % testData.rows;
		Mat randShowImg = testData.row(imgIdx).reshape(0, 57);
		float pd = model->predict(randShowImg.reshape(0, 1));
		std::cout << "Trial " + std::to_string(i) << " "
			<< (std::abs(pd - testLabels.at<int>(imgIdx)) <= FLT_EPSILON ? "Correct" : "Incorrect")
			<< std::endl;
		randShowImg.convertTo(randShowImg, CV_8U);
		img_show("Trial " + std::to_string(i) + " Predicted as " + std::to_string((int)pd), randShowImg, 300, 360);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}