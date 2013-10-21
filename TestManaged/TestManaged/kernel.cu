#include "cuda_runtime.h"
#include <stdio.h>
 
extern "C"  
{
	__global__ void kernel(int a, int b, int *c)
	{
		*c = (a + b)*(a + b);
	}
}
 
int main()
{
    return 0;
}