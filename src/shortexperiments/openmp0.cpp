#include <iostream>
#include "helper_timer.h"

void simple(int n, float *a, float *b, unsigned threadsNum)
{
	int i;

#pragma omp parallel for num_threads(threadsNum), schedule (static)
    for (i=0; i < n; i++) {
    	float sum=0.0;
    	for (int j = i; j < n; ++j)
    	{
    		sum += a[j];
    	}

    	float tmp = (a[i]*a[i] + b[i]*b[i]);
		float k = a[i]*b[i] + a[i];
		b[i] = (tmp*5*k) + sum;
    }
}

template<typename T>
void printArray(T* a, int N) {
	std::cout << "Array:" << std::endl;
	for (int i = 0; i < N; ++i)
	{
		std::cout << "\t" << a[i] << std::endl;
	}
	std::cout << "***" << std::endl << std::endl;
}

template<typename T>
void initArray(T* a, int N, float v) {
	float v_ = v;
	for (int i = 0; i < N; ++i)
		a[i] = v_++;
}



int main() {
	using namespace std;
	
	const int N = 100000;
	const int K = 10;
	float a[N];
	float b[N];

	initArray<float>(a, N, 3.0f);
	initArray<float>(b, N, 5.0f);
	
	
	
	unsigned threadsNum = 1;
	for (int i = 0; i < K; ++i)
	{
		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);

		simple(N, a, b, threadsNum);

		sdkStopTimer(&timer);
		float execution_time = sdkGetTimerValue(&timer);
		cout << "THREAD_NUM: " << threadsNum << ", TIME: " << execution_time << endl;

		threadsNum *=2;
	}

	return 0;
}