#include "cuda_runtime.h"
#include <stdio.h>
 
extern "C"  
{
	__global__ void kernel(int *m, int v)
	{
		int index = threadIdx.x + (blockDim.x * blockIdx.x);
		m[index] = m[index] + v;
	}
}
 
int main()
{
    return 0;
}