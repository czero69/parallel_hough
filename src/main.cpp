#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "EyeDescriptor.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	/*if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}*/

#ifdef _DEBUG
	cout << "Debug mode" << std::endl;
#endif

	Mat image;
	image = imread("/Users/Kadlubek47/Programming/School/Elka2015Z/PORR/project/src/photos/1.png",-1); // Read the file, -1 = 'image type as is'

	cv::resize(image, image, Size(), 0.67, 0.67, INTER_CUBIC);


	//resize(image, image, Size(), 0.5, 0.5, INTER_CUBIC);
	//@TODO camera stre, usb 3.0 driver 120fps
	//@TODO testUnit class

	//@TODO add support for 16UC1 uchar is only for 8UC1

	//image.convertTo(image, CV_BGRA2GRAY); //16UC1

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cout << "mat type is" << image.type(); //http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html

	EyeDescriptor rightEye(image);

	Mat image2;

	double t = (double)getTickCount();
	rightEye.HoughCircle(image, image2);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "/nTimes passed in seconds: " << t << endl;
	//cv::waitKey(0); // Wait for a keystroke in the window

	cv::cvtColor(image, image, CV_GRAY2RGB);
	cout << "mat type is" << image.type();
	cv::circle(image, cv::Point(rightEye.x_maxval, rightEye.y_maxval), rightEye.r_maxval, cv::Scalar(50,200,0));

	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", image);
	cv::imwrite("/Users/Kadlubek47/Programming/School/Elka2015Z/PORR/project/src/wyniki/7.png", image);
	cv::waitKey(0);

	return 0;
}