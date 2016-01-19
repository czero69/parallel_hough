#include "EyeDescriptor.h"
#include "timer/helper_timer.h"
#include <iostream>

#define M_PI 3.14159265358979323846



EyeDescriptor::EyeDescriptor(const cv::Mat& mat)
{
	this->width = mat.cols;
	this->height = mat.rows;
	this->ostepnumb = 256;
	this->localGradient_angles  = (int*)calloc(height*width,sizeof(int)); //a lot of mem

	rdelta = 2;
	omax = M_PI;
	omin = -M_PI;
	owidth = ostepnumb;
	odelta = (omax - omin) / ostepnumb;
	
	rmin = -sqrt(float(pow(width, 2) + pow(height, 2)))/2;
	rmax =  sqrt(float(pow(width, 2) + pow(height, 2)))/2;
	rstepnumb = int(rmax - rmin)/2/rdelta;
	rwidth = rstepnumb;

	si = (float*)malloc(ostepnumb*sizeof(float));
	ci = (float*)malloc(ostepnumb*sizeof(float));


	for (int i = 0; i < ostepnumb; i++)
	{
		si[i] = sin(omin + i*odelta);
		ci[i] = cos(omin + i*odelta);
	}

	houghmaxval = 0;
	x_maxval = 0;
	y_maxval = 0;
	r_maxval = 0;

	accummulator = (int*)calloc(height*width*rstepnumb, sizeof(int)); //it is a huge amount of data space.. check how much mem did you alloc?
}


EyeDescriptor::~EyeDescriptor()
{
	free(si);
	free(ci);
	free(localGradient_angles);
	free(accummulator);
}

void EyeDescriptor::HoughCircle(const cv::Mat& src, cv::Mat& dst){
	src.convertTo(dst, CV_8UC1);
	cv::GaussianBlur(dst, dst, cv::Size(15, 15), 25, 25);

#ifdef _DEBUG
	cv::imshow("blur", dst); // Show our image inside it.
	cv::imwrite("wyniki\\1.png", dst);
#endif

	gradientsobel(dst);

	mythreshold(dst, dst, 60);

#ifdef _DEBUG
	cv::imshow("threshold", dst); // Show our image inside it.
	cv::imwrite("wyniki\\4.png", dst);
#endif

	int dilation_size1 = 3;
	int erosion_size1 = 3;

	cv::Mat element_dil1 = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * dilation_size1 + 1, 2 * dilation_size1 + 1),
		cv::Point(dilation_size1, dilation_size1));
	cv::Mat element_erode1 = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size1 + 1, 2 * erosion_size1 + 1),
		cv::Point(erosion_size1, erosion_size1));

	cv::erode(dst, dst, element_erode1);
	cv::dilate(dst, dst, element_dil1);

	cv::dilate(dst, dst, element_dil1);
	cv::erode(dst, dst, element_erode1);

#ifdef _DEBUG 
	cv::imshow("Morpho open and close", dst); // Show our image inside it.
	cv::imwrite("wyniki\\5.png", dst);
#endif

	cv::Canny(dst, dst, 60, 200); //@TODO replace, gradient calc is already here!

#ifdef _DEBUG 
	cv::imshow("Canny", dst); // Show our image inside it.
	cv::imwrite("wyniki\\5.png", dst);
#endif

	mainHough(dst);

#ifdef _DEBUG
	cv::circle(dst, cv::Point(x_maxval, y_maxval), r_maxval, cv::Scalar(50, 200, 0));
	cv::imshow("solution circle", dst); // Show our image inside it.
	cv::imwrite("wyniki\\6.png", dst);
#endif

}

void EyeDescriptor::gradientsobel(const cv::Mat& mat){
	float threshold = 2;

#ifdef _DEBUG
	cv::Mat matGX = cv::Mat(mat.rows, mat.cols, CV_8UC1, cvScalar(0));
	cv::Mat matGY = cv::Mat(height, width, CV_8UC1, cvScalar(0));
#endif

	for (int y = 1; y < mat.rows - 1; y++)
		for (int x = 1; x < mat.cols - 1; x++)
		{
			float convF[3] = { +0.5, 0, -0.5 }; //flipped, poprawiono blad konwolucja jest teraz z lustrzanym odbiciem kernela

			float GX = convF[0] * mat.at<uchar>(y, x - 1) + convF[1] * mat.at<uchar>(y, x) + convF[2] * mat.at<uchar>(y, x + 1);
			float GY = convF[0] * mat.at<uchar>(y - 1, x) + convF[1] * mat.at<uchar>(y, x) + convF[2] * mat.at<uchar>(y + 1, x);

#ifdef _DEBUG
			matGX.at<uchar>(y, x) = uchar(std::abs(GX) * 10);
			matGY.at<uchar>(y, x) = uchar(std::abs(GY) * 10);
#endif

			float G = std::abs(GX) + std::abs(GY); //edge strenght 
			float angle = atan2(GY, GX); //return value [-pi,+pi]

			int quantized_angle = int((angle + M_PI) / (2 * M_PI) * ostepnumb + 0.5);
			if (quantized_angle == ostepnumb) 
				quantized_angle = ostepnumb - 1;
			localGradient_angles[x + y*width] = quantized_angle;
		}

#ifdef _DEBUG
	cv::imshow("grad X", matGX); // Show our image inside it.
	cv::imshow("grad Y", matGY); // Show our image inside it.
	cv::imwrite("wyniki\\2.png", matGX);
	cv::imwrite("wyniki\\3.png", matGY);
#endif
}

void EyeDescriptor::mythreshold(cv::Mat src, cv::Mat &dst, int thresh)
{

	src.convertTo(dst,CV_8UC1);
	uchar* pixelPtr_K = (uchar*)dst.data;
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{
			if (thresh < pixelPtr_K[i*dst.cols + j])
				pixelPtr_K[i*dst.cols + j] = 0;
			else
				pixelPtr_K[i*dst.cols + j] = 255;
		}
}