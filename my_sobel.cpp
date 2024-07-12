#pragma comment(lib, "opencv_world460d.lib")
#include"utilities.hpp"

using namespace cv;

double scale = 1;
unsigned int ksize = 3;
double delta = 0;
int ddepth = CV_16S;

int my_sobel() {
	auto image = imread("eyjafjalla.jpg", IMREAD_COLOR);
    Mat src, src_gray, grad;
    for (;;)
    {
        //![reduce_noise]
        // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
        //![reduce_noise]

        //![convert_to_gray]
        // Convert the image to grayscale
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
        //![convert_to_gray]

        //![sobel]
        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        /// Gradient X
        Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);

        /// Gradient Y
        Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
        //![sobel]

        //![convert]
        // converting back to CV_8U
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        //![convert]

        //![blend]
        /// Total Gradient (approximate)
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        //![blend]

        //![display]
        //imshow("Grad", grad);
        img_show("Grad X", grad_x);
        img_show("Grad Y", grad_y);
        img_show("Grad", grad);
        char key = (char)waitKey(0);
        //![display]

        if (key == 27)
        {
            return EXIT_SUCCESS;
        }

        if (key == 'k' || key == 'K')
        {
            ksize = ksize < 30 ? ksize + 2 : -1;
        }

        if (key == 's' || key == 'S')
        {
            scale++;
        }

        if (key == 'd' || key == 'D')
        {
            delta++;
        }

        if (key == 'r' || key == 'R')
        {
            scale = 1;
            ksize = -1;
            delta = 0;
        }
    }

	return 0;
}