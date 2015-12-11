#include "EyeDescriptor.h"

#define M_PI 3.14159265358979323846

EyeDescriptor::EyeDescriptor(const cv::Mat& mat)
{
	this->width = mat.cols;
	this->height = mat.rows;
	this->ostepnumb = 256;
	this->localGradient_angles  = (int*)calloc(height*width,sizeof(int)); //a lot of mem

	omax = M_PI;
	omin = -M_PI;
	owidth = ostepnumb;
	odelta = (omax - omin) / ostepnumb;
	
	rmin = -sqrt(double(pow(width, 2) + pow(height, 2)))/2;
	rmax = sqrt(double(pow(width, 2) + pow(height, 2))) /2;
	rstepnumb = int(rmax - rmin)/4;
	rwidth = rstepnumb;
	rdelta = ((rmax - rmin) / rstepnumb);

	//anglesBuffer = (int*)malloc(height*width*sizeof(int));
	si = (double*)malloc(ostepnumb*sizeof(double));
	ci = (double*)malloc(ostepnumb*sizeof(double));


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

	//@TODO add support for 16UC1, uchar is only for 8UC1
	//upd 9.12.15 bo z tej kamerki co zapodaje obrazki to sa one pewnie 16UC1

	//src.copyTo(dst);

	//cv::Mat mat_cpy; //test
	//src.convertTo(mat_cpy, CV_8UC1); //test

	//cv::Mat mat_cpy = src.clone();
	//mat_cpy = src.clone();
	
	src.convertTo(dst, CV_8UC1);

	//dst = src.clone();
	//GradientSobel(mat_cpy);

	cv::GaussianBlur(dst, dst, cv::Size(15, 15), 25, 25);

	#ifdef _DEBUG
	cv::imshow("blur", dst); // Show our image inside it.
	cv::imwrite("wyniki\\1.png", dst);
	#endif

	gradientsobel(dst);
	//cv::threshold(dst, dst, 60, 255, cv::THRESH_BINARY_INV);
	mythreshold(dst, dst, 60);

	#ifdef _DEBUG
	cv::imshow("threshold", dst); // Show our image inside it.
	cv::imwrite("wyniki\\4.png", dst);
	//cv::waitKey(0); // Wait for a keystroke in the window
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
	//cv::waitKey(0); // Wait for a keystroke in the window
	#endif


	//#ifdef _DEBUG
	//cv::dilate(mat_cpy, mat_cpy, element_dil1);
	//cv::erode(mat_cpy, mat_cpy, element_erode1);
	//cv::imshow("test dilate", mat_cpy);
	//#endif

	//gradientsobel(dst);

	cv::Canny(dst, dst, 60, 200); //@TODO replace, gradient calc is already here!

	#ifdef _DEBUG 
	cv::imshow("Canny", dst); // Show our image inside it.
	cv::imwrite("wyniki\\5.png", dst);
	//cv::waitKey(0); // Wait for a keystroke in the window
	#endif

	for (int y = 1; y < height-1; y++)
		for (int x = 1; x < width-1; x++)
		{
		if (dst.at<uchar>(y, x) == 255)
		{

		int q_angle = localGradient_angles[x + y*width];

		//double angle = (quantized_angle / ostepnumb) * 2 * M_PI - M_PI; //return angle value, the circle is at this line

		for (double r = rmin; r < rmax; r+=rdelta)
		{
			int eps = 40;
			if (std::abs(r) < eps)
				continue;

			int ri = int(((r - rmin) / (rmax - rmin))*rstepnumb);
			if (ri == rstepnumb) ri = rstepnumb - 1; //small chance for drawing rmax and getting out of bounds

			int x0 = int(x - r*ci[q_angle] + 0.5);
			int y0 = int(y - r*si[q_angle] + 0.5);

			if (!(x0>=0 && x0 < width && y0>=0 && y0 < height))
				continue;
 			int tmp = ++accummulator[x0 + y0*width + ri*height*width];
			if (tmp > houghmaxval){
				houghmaxval = tmp;
				x_maxval = x0;
				y_maxval = y0;
				r_maxval = int(std::abs(r)+0.5);
			}
		}

		}

		}

	#ifdef _DEBUG
	cv::circle(dst, cv::Point(x_maxval, y_maxval), r_maxval, cv::Scalar(50, 200, 0));
	cv::imshow("solution circle", dst); // Show our image inside it.
	cv::imwrite("wyniki\\6.png", dst);
	#endif
	//cv::waitKey(0); // Wait for a keystroke in the window
}

void EyeDescriptor::gradientsobel(const cv::Mat& mat){

	//@TODO add support for 16UC1 uchar, now it is only for 8UC1

	double threshold = 2;
	//if debug
	#ifdef _DEBUG
	cv::Mat matGX = cv::Mat(mat.rows, mat.cols, CV_8UC1, cvScalar(0));
	cv::Mat matGY = cv::Mat(height, width, CV_8UC1, cvScalar(0));
	#endif

	//std::vector<double> anglesVote(N_X_BLOCKS*N_Y_BLOCKS, 0);

	//int * anglesBuffer; 
	//anglesBuffer = (int*)malloc(height*width*sizeof(int));

	for (int y = 1; y < mat.rows - 1; y++)
		for (int x = 1; x < mat.cols - 1; x++)
		{
		//double gradX[3];
		//double gradY[3];

		double convF[3] = { +0.5, 0, -0.5 }; //flipped, poprawiono blad konwolucja jest teraz z lustrzanym odbiciem kernela

		double GX = convF[0] * mat.at<uchar>(y, x - 1) + convF[1] * mat.at<uchar>(y, x) + convF[2] * mat.at<uchar>(y, x + 1);
		double GY = convF[0] * mat.at<uchar>(y - 1, x) + convF[1] * mat.at<uchar>(y, x) + convF[2] * mat.at<uchar>(y + 1, x);
		//if debug

		#ifdef _DEBUG
		matGX.at<uchar>(y, x) = uchar(std::abs(GX) * 10);
		matGY.at<uchar>(y, x) = uchar(std::abs(GY) * 10);
		#endif

		double G = std::abs(GX) + std::abs(GY); //edge strenght 
		//if (G < threshold)
		//	continue;
		 
		double angle = atan2(GY, GX); //return value [-pi,+pi]
		// quantize angle to the range 0..ostepnumb-1

		//angle += odelta / 2;
		//if (angle > M_PI)
		//	angle -= M_PI;

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
	//cv::waitKey(0); // Wait for a keystroke in the window

}

void EyeDescriptor::mythreshold(cv::Mat src, cv::Mat &dst, int thresh)
{
	//dst = src.clone();
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