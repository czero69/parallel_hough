#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

class EyeDescriptor
{
public:
	unsigned threadsNum;
	float lastExecutionTime;

	EyeDescriptor(const cv::Mat& mat);
	~EyeDescriptor();

	void HoughCircle(const cv::Mat& src, cv::Mat& dst);

	int x_maxval;
	int y_maxval;
	int r_maxval;

	int* accummulator;

private:

	void gradientsobel(const cv::Mat& mat);
	void mythreshold(cv::Mat src, cv::Mat &dst, int thresh);
	int width;
	int height;
	int * localGradient_angles;

	float rmin;
	float rmax;
	int rstepnumb;
	int rwidth;
	float rdelta;

	float omin;
	float omax; 	// max angle value
	int ostepnumb; 	//how much step does the hough transform angle have
	int owidth;
	float odelta;

	float* si,* ci;

	int houghmaxval;

	void mainHough(cv::Mat& dst);

};

