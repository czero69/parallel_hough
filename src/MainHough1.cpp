#include "EyeDescriptor.h"
#include "timer/helper_timer.h"
#include <iostream>
#include <omp.h>


void EyeDescriptor::mainHough(cv::Mat& dst) {
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	//pixels which should be considered
	std::vector<std::pair<int, int>> edgesIdx;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (dst.at<uchar>(y, x) == 255)
			{
				edgesIdx.push_back( std::pair<int, int>(x, y) );
			}
	
	int NPixelsEdges = (int) edgesIdx.size();
	int* pixelsDone = new int[threadsNum]();

	sdkStopTimer(&timer);

#pragma omp parallel for num_threads(this->threadsNum), schedule (static)
	for (int ipixel = 0; ipixel < NPixelsEdges; ++ipixel)
	{
		++pixelsDone[omp_get_thread_num()];

		int x = edgesIdx[ipixel].first;
		int y = edgesIdx[ipixel].second;
		
		//gradient angle?
		int q_angle = localGradient_angles[x + y * width];
		
		//we check for each radius
		//	from rmin to rmax
		for (double r = rmin; r < rmax; r += rdelta)
		{
			int eps = 40;
			if (std::abs(r) < eps)
				continue;
			
			int ri = int(((r - rmin) / (rmax - rmin))*rstepnumb);
			if (ri == rstepnumb)
				ri = rstepnumb - 1; //small chance for drawing rmax and getting out of bounds
			
			int x0 = int(x - r*ci[q_angle] + 0.5);
			int y0 = int(y - r*si[q_angle] + 0.5);
			
			if (!(x0>=0 && x0 < width && y0>=0 && y0 < height))
				continue;
			
#pragma omp atomic update
			++(accummulator[x0 + y0*width + ri*height*width]);
		}
	}

	// sdkStopTimer(&timer);

	int max = 0; 
	x_maxval = 0;
	y_maxval = 0;
	r_maxval = 0;
	for (int ix = 0; ix < width; ++ix)
		for (int iy = 0; iy < height; ++iy)
			for (int ir = 0; ir < rstepnumb; ++ir)
			{
				if (accummulator[ix + iy*width + ir*height*width] > max) {
					max = accummulator[ix + iy*width + ir*height*width];
					x_maxval = ix; 
					y_maxval = iy;
					r_maxval = std::abs(ir*(rmax - rmin)/rstepnumb + rmin);
				}
			}
	houghmaxval = max;

	for (int i = 0; i < threadsNum; ++i)
	{
		std::cout << "\tPixelsMade by thread" << i << ": " << pixelsDone[i] << std::endl;
	}
	
	// sdkStopTimer(&timer);
	float execution_time = sdkGetTimerValue(&timer);
	this->lastExecutionTime = execution_time;
	std::cout << "Main loop, THREADS_NUM: " << threadsNum << " TIME: " << execution_time << "[ms]" << std::endl << std::endl;
}