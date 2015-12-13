/*
	//resize(image, image, Size(), 0.5, 0.5, INTER_CUBIC);
	//@TODO camera stre, usb 3.0 driver 120fps
	//@TODO testUnit class
	//@TODO add support for 16UC1 uchar is only for 8UC1
	//image.convertTo(image, CV_BGRA2GRAY); //16UC1
 
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "EyeDescriptor.h"

using namespace cv;
using namespace std;

#define __OPENMP_MEASURE__
//#define __CHECK_CORRECTNESS__



int mainRun(unsigned threadsNum, std::string imageInName, float& lastExecutionTime) {

#ifdef _DEBUG
	cout << "Debug mode" << std::endl;
#endif

	Mat image;
	image = imread("/Users/Kadlubek47/Programming/School/Elka2015Z/PORR/project/dataInOut/in/1.png",-1); // Read the file, -1 = 'image type as is'

	cv::resize(image, image, Size(), 1, 1, INTER_CUBIC);

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

#ifndef __OPENMP_MEASURE__
	cout << "mat type is" << image.type(); //http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html
#endif

	EyeDescriptor rightEye(image);

	Mat image2;

#ifndef __OPENMP_MEASURE__
	double t = (double)getTickCount();
#endif

	rightEye.threadsNum = threadsNum;
	rightEye.HoughCircle(image, image2);
	lastExecutionTime = rightEye.lastExecutionTime;

#ifndef __OPENMP_MEASURE__
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "/nTimes passed in seconds: " << t << endl;
#endif

	cv::cvtColor(image, image, CV_GRAY2RGB);
#ifndef __OPENMP_MEASURE__
	cout << "mat type is" << image.type();
#endif
	cv::circle(image, cv::Point(rightEye.x_maxval, rightEye.y_maxval), rightEye.r_maxval, cv::Scalar(50,200,0));

#ifndef __APPLE__   
	cv::imwrite("/Users/Kadlubek47/Programming/School/Elka2015Z/PORR/project/dataInOut/out/7.png", image);
#endif

#ifdef __CHECK_CORRECTNESS__
	cv::imshow("Display window", image);
	cv::waitKey(0);
#endif

	return 0;
}



int main(int argc, char** argv)
{
	std::string filename("name");
	int returnCode=0;

	int threadsNum = 1;
	const int runperparameters = 1;
	const int differentThreadsNumRunCount = 5;
	
	float lastExecutionTime;
	float executionTimes[differentThreadsNumRunCount][runperparameters];
	for (int i = 0; i < differentThreadsNumRunCount; ++i)
	{
		for (int j = 0; j < runperparameters; ++j) {
			mainRun(threadsNum, filename, lastExecutionTime);
			executionTimes[i][j] = lastExecutionTime;
		}
		threadsNum *= 2;
	}

	for (int i = 0; i < differentThreadsNumRunCount; ++i)
		std::cout << executionTimes[i][0] << " " << std::endl;

	return 0;
}