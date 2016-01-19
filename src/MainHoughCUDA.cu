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
		
		atomicAdd(&accummulator[x0 + y0*width + ri*height*width], 1);
		// ++accummulator[x0 + y0*width + ri*height*width];
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
	checkCudaErrors(cudaMalloc((void **) &d_edgesIdx, 							SIZEI(NPixelsEdges*2)));
	checkCudaErrors(cudaMalloc((void **) &d_localGradient_angles, 				SIZEI(height*width)));
	checkCudaErrors(cudaMalloc((void **) &d_accummulator, 						SIZEI(height*width*rstepnumb)));
	checkCudaErrors(cudaMalloc((void **) &d_si, 								SIZEF(ostepnumb)));
	checkCudaErrors(cudaMalloc((void **) &d_ci, 								SIZEF(ostepnumb)));
	checkCudaErrors(cudaDeviceSynchronize());

	/*
		Transfer data from CPU to GPU
	*/
	checkCudaErrors(cudaMemcpy(d_edgesIdx, edgesIdx, 							SIZEI(NPixelsEdges*2), 			cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_localGradient_angles, localGradient_angles, 	SIZEI(height*width), 			cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_accummulator, accummulator, 					SIZEI(height*width*rstepnumb), 	cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_accummulator, 0, SIZEI(height*width*rstepnumb)));
	checkCudaErrors(cudaMemcpy(d_si, si, 										SIZEF(ostepnumb), 				cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ci, ci, 										SIZEF(ostepnumb), 				cudaMemcpyHostToDevice));

	cudaEvent_t start, stop; float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	/*
		call KERNEL
	*/
	int K = 512;
	kernelCPU <<<(NPixelsEdges+K-1)/K, K>>> (
		height,
		width,

		d_edgesIdx,
		NPixelsEdges,

		d_localGradient_angles,

		rmin,
		rmax,
		rdelta,
		rstepnumb,

		d_accummulator,

		d_si,
		d_ci
	);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaDeviceSynchronize());

	/*
		Transfer data from GPU to CPU
	*/
	// checkCudaErrors(cudaMemcpy(edgesIdx, d_edgesIdx,  							SIZEI(NPixelsEdges*2), 			cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(localGradient_angles, d_localGradient_angles,  	SIZEI(height*width), 			cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(accummulator, d_accummulator,  					SIZEI(height*width*rstepnumb), 	cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(si, d_si,  										SIZEF(ostepnumb), 				cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(ci, d_ci,  										SIZEF(ostepnumb), 				cudaMemcpyDeviceToHost));

	/*
		Synchronize
	*/
	checkCudaErrors(cudaDeviceSynchronize());

	sdkStopTimer(&timer);
	float execution_time = sdkGetTimerValue(&timer);
	std::cout << "Main loop, TIME: " << execution_time << "[ms]" << std::endl;

	/*
		compute maxval
	*/
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

	std::cout << "Main loop, THREADS_NUM: " << NPixelsEdges << " TIME: " << execution_time << "[ms]" << std::endl << std::endl;
	std::cout << "TIME GPU only: " << elapsedTime << std::endl;
}