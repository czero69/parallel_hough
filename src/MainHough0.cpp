#include "EyeDescriptor.h"
#include "timer/helper_timer.h"
#include <iostream>



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
	for (int ipixel = 0; ipixel < NPixelsEdges; ++ipixel)
	{
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
			
			int tmp = ++accummulator[x0 + y0*width + ri*height*width];
			
			if (tmp > houghmaxval){
				houghmaxval = tmp;
				x_maxval = x0;
				y_maxval = y0;
				r_maxval = int(std::abs(r)+0.5);
			}
		}
	}
	
	sdkStopTimer(&timer);
	float execution_time = sdkGetTimerValue(&timer);
	std::cout << "Main loop, TIME: " << execution_time << "[ms]" << std::endl;
}