#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace ml;

void my_knn_digit() {
	Mat img = imread("digits.png", IMREAD_GRAYSCALE);
	int b = 20;
	int m = img.rows / b;   //ԭͼΪ1000*2000
	int n = img.cols / b;   //�ü�Ϊ5000��20*20��Сͼ��
	Mat data, labels;   //��������

	for (int i = 0; i < n; i++){
		int offsetCol = i * b; //���ϵ�ƫ����
		for (int j = 0; j < m; j++){
			int offsetRow = j * b;  //���ϵ�ƫ����
								  //��ȡ20*20��С��
			Mat tmp;
			img(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			//reshape  0��ͨ������  �������֣���ʾҪ���õ�ͨ����
			//reshape  ��ʾ�����������������Ϊ0�����ʾ����ԭ���������䣬�������Ϊ�������֣���ʾҪ���õ�����
			data.push_back(tmp.reshape(0, 1));  //���л��������������
			labels.push_back((int)j / 5);  //��Ӧ�ı�ע
		}
	}

	data.convertTo(data, CV_32F); //uchar��ת��Ϊcv_32f
	int samplesNum = data.rows;
	int trainNum = 500;
	Mat trainData, trainLabels;
	trainData = data(Range(0, trainNum), Range::all());   //ǰ3000������Ϊѵ������
	trainLabels = labels(Range(0, trainNum), Range::all());

	//ʹ��KNN�㷨
	int K = 5;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	auto model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
	//Ԥ�����
	double train_hr = 0, test_hr = 0;
	Mat response;
	//�������
	for (int i = 0; i < samplesNum; i++)
	{
		Mat sample = data.row(i);
		float r = model->predict(sample);   //�������н���Ԥ��
											//Ԥ������ԭ�����ȣ����Ϊ1������Ϊ0
		r = std::abs(r - labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < trainNum)
			train_hr += r;  //�ۻ���ȷ��
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
	//�ر�OpenCV����̨���
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	Mat trainData, trainLabels;
	Mat testData, testLabels;
	load_Olivetti_Face(trainData, trainLabels, testData, testLabels);

	//ʹ��KNN�㷨
	int K = 5;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	auto model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
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