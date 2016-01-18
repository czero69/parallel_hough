#include "EyeDescriptor.h"
#include "timer/helper_timer.h"
#include <iostream>



void kernelCPU(
	int ipixel,

	int height,
	int width,

	int* edgesIdx, 				/* READONLY */
	int NPixelsEdges,

	int* localGradient_angles, 	/* READONLY */

	float rmin,
	float rmax,
	float rdelta,
	int rstepnumb,

	int* accummulator, 			/* READ WRITE */

	int* maxval,  				/* READ WRITE */ 
								/* CASTED */
								/*hough, x, y, r 4floats*/

	float* si,
	float* ci
) 
{
	// int ipixel = threadIdx.x + blockIdx.x*blockDim.x;
	if ( ipixel >= NPixelsEdges ) return; 



	int x = edgesIdx[ipixel*2];
	int y = edgesIdx[ipixel*2+1];

	//gradient angle
	int q_angle = localGradient_angles[x + y * width];

	//we check for each radius
	//	from rmin to rmax
	for (float r = rmin; r < rmax; r += rdelta)
	{
		int eps = 40;
		if (std::abs(r) < eps)
			continue;
		
		int ri = int(((r - rmin) / (rmax - rmin))*rstepnumb);
		if (ri == rstepnumb)
			ri = rstepnumb - 1;
		
		int x0 = int(x - r*ci[q_angle] + 0.5);
		int y0 = int(y - r*si[q_angle] + 0.5);
		
		if (!(x0>=0 && x0 < width && y0>=0 && y0 < height))
			continue;
		
		int tmp = ++accummulator[x0 + y0*width + ri*height*width];
		
		if (tmp > maxval[0]){
			maxval[0] = tmp;
			maxval[1] = x0;
			maxval[2] = y0;
			maxval[3] = int(std::abs(r)+0.5);
		}
	}
}



void EyeDescriptor::mainHough(cv::Mat& dst) {
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	int NPixelsEdges = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (dst.at<uchar>(y, x) == 255)
				++NPixelsEdges;

	int* edgesIdx = new int[NPixelsEdges*2];
	int iedge = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (dst.at<uchar>(y, x) == 255)
			{
				edgesIdx[iedge*2] = x;
				edgesIdx[iedge*2+1] = y;
				++iedge;
			}


	/*
		Prepare inputs
			maxval
	*/
	int maxval[4];
	maxval[0]=houghmaxval;
	maxval[1]=x_maxval;
	maxval[2]=y_maxval;
	maxval[3]=r_maxval;


	/*
		call KERNEL
	*/
	for (int ipixel = 0; ipixel < NPixelsEdges; ++ipixel)
	{
		kernelCPU(
			ipixel,

			height,
			width,

			edgesIdx,
			NPixelsEdges,

			localGradient_angles,

			rmin,
			rmax,
			rdelta,
			rstepnumb,

			accummulator,
			maxval,

			si,
			ci
		);
	}
	

	/*
		Cast outputs to curr data model;
	*/
	houghmaxval=maxval[0];
	x_maxval=maxval[1];
	y_maxval=maxval[2];
	r_maxval=maxval[3];


	sdkStopTimer(&timer);
	float execution_time = sdkGetTimerValue(&timer);
	std::cout << "Main loop, TIME: " << execution_time << "[ms]" << std::endl;
}