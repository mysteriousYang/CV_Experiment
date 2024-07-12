#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;

void _cannyThreshold(int, void*)
{

    blur(src_gray, detected_edges, Size(3, 3));

    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);

    //! img_show("Edge Map", dst, 1017, 720);
    img_show("Edge Map", dst, 480, 720);
}


void my_canny()
{
    src = imread("_MG_7526.jpg", IMREAD_COLOR); // Load an image

    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

    namedWindow("Edge Map", WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
    resizeWindow("Edge Map", 480, 720);
    //resizeWindow("Edge Map", 1017, 720);

    createTrackbar("Min Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, _cannyThreshold);

    _cannyThreshold(0, 0);
    waitKey(0);
    return;
}