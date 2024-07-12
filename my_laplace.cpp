#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

int my_laplace() {
    //![variables]
    // Declare the variables we are going to use
    Mat src, src_f, dst, dst_f;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    //![variables]

    //![load]
    const char* imageName = "eyjafjalla_noised.png";

    src = imread(samples::findFile(imageName), IMREAD_GRAYSCALE);
    src_f = imread(samples::findFile(imageName), IMREAD_GRAYSCALE); // Load an image

    //![reduce_noise]
    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(src_f, src_f, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //![reduce_noise]

    /// Apply Laplace function
    Mat abs_dst, abs_dst_f;
    //![laplacian]
    Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    Laplacian(src_f, dst_f, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    //![laplacian]

    //![convert]
    // converting back to CV_8U
    convertScaleAbs(dst, abs_dst);
    convertScaleAbs(dst_f, abs_dst_f);
    //![convert]

    //![display]
    img_show("Laplace", abs_dst);
    img_show("Filtered Laplace", abs_dst_f);
    waitKey(0);
    //![display]

	return 0;
}