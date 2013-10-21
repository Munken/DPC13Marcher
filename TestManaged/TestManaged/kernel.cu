#include "cuda_runtime.h"
#include <stdio.h>
 
__global__ void kernel(int a, int b, int *c)
{
    *c = (a + b)*(a + b);
}
 
int main()
{
    return 0;
}