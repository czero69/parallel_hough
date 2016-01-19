#include "EyeDescriptor.h"
#include "timer/helper_timer.h"
#include <iostream>

//CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>  


#define SIZEF(x) (sizeof(float)*(x))
#define SIZEI(x) (sizeof(int)*(x))


__global__
static void kernelCPU(
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
	int ipixel = threadIdx.x + blockIdx.x*blockDim.x;
	if ( ipixel >= NPixelsEdges ) return; 



	int x = edgesIdx[ipixel*2];
	int y = edgesIdx[ipixel*2+1];

	//gradient angle
	int q_angle = localGradient_angles[x + y * width];

	//we check for each radius
	//	from rmin to rmax
	for (double r = rmin; r < rmax; r += rdelta)
	{
		int eps = 40;
		if (abs(r) < eps)
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
			maxval[3] = int(abs(r)+0.5);
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
		Cast data to flat memory
	*/
	int maxval[4];
	maxval[0]=houghmaxval;
	maxval[1]=x_maxval;
	maxval[2]=y_maxval;
	maxval[3]=r_maxval;


	//
	//	CUDA init
	//
	int bytesAllocated = 
		((SIZEI(NPixelsEdges))) +
		((SIZEI(height*width))) +
		((SIZEI(height*width*rstepnumb))) +
		((SIZEI(4))) +
		((SIZEF(ostepnumb))) +
		((SIZEF(ostepnumb)));
	std::cout << "Memory to allocate on GPU is: " << bytesAllocated/1024/1024 << "[MB]" << std::endl;

	//CUDA memory
	int* d_edgesIdx;
	int* d_localGradient_angles;
	int* d_accummulator;
	int* d_maxval;
	float* d_si;
	float* d_ci;

	/*
		Alloc memory on GPU
	*/
	checkCudaErrors(cudaMalloc((void **) &d_edgesIdx, 							SIZEI(NPixelsEdges)));
	checkCudaErrors(cudaMalloc((void **) &d_localGradient_angles, 				SIZEI(height*width)));
	checkCudaErrors(cudaMalloc((void **) &d_accummulator, 						SIZEI(height*width*rstepnumb)));
	checkCudaErrors(cudaMalloc((void **) &d_maxval, 							SIZEI(4)));
	checkCudaErrors(cudaMalloc((void **) &d_si, 								SIZEF(ostepnumb)));
	checkCudaErrors(cudaMalloc((void **) &d_ci, 								SIZEF(ostepnumb)));

	/*
		Transfer data from CPU to GPU
	*/
	checkCudaErrors(cudaMemcpy(d_edgesIdx, edgesIdx, 							SIZEF(NPixelsEdges), 			cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_localGradient_angles, localGradient_angles, 	SIZEF(height*width), 			cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_accummulator, accummulator, 					SIZEF(height*width*rstepnumb), 	cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_maxval, maxval, 								SIZEF(4), 						cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_si, si, 										SIZEF(ostepnumb), 				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ci, ci, 										SIZEF(ostepnumb), 				cudaMemcpyHostToDevice));

	/*
		call KERNEL
	*/
	int K = 512;
	kernelCPU <<<(NPixelsEdges+K-1)/K, K>>> (
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

	/*
		Transfer data from GPU to CPU
	*/
	checkCudaErrors(cudaMemcpy(edgesIdx, d_edgesIdx,  							SIZEF(NPixelsEdges), 			cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(localGradient_angles, d_localGradient_angles,  	SIZEF(height*width), 			cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(accummulator, d_accummulator,  					SIZEF(height*width*rstepnumb), 	cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(maxval, d_maxval,  								SIZEF(4), 						cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(si, d_si,  										SIZEF(ostepnumb), 				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ci, d_ci,  										SIZEF(ostepnumb), 				cudaMemcpyDeviceToHost));

	/*
		Synchronize
	*/
	checkCudaErrors(cudaDeviceSynchronize());

	/*
		Cast data from flat memory to C++ classess model;
	*/
	houghmaxval=maxval[0];
	x_maxval=maxval[1];
	y_maxval=maxval[2];
	r_maxval=maxval[3];


	sdkStopTimer(&timer);
	float execution_time = sdkGetTimerValue(&timer);
	std::cout << "Main loop, TIME: " << execution_time << "[ms]" << std::endl;
}