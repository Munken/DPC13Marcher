#include "cuda_runtime.h"
#include <stdio.h>
 
extern "C"  
{
	__global__ void kernel(int *m, int v)
	{
     m[threadIdx.x + (blockDim.x * blockIdx.x)] = v;
	}
}
 
int main()
{
    return 0;
}