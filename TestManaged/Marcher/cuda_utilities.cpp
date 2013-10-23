#include "cuda_utilities.h"
#include "error.h"

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 */
void CHECK_FOR_CUDA_ERROR_FUNCTION(const char* file, const char* line) {
	cudaError_t errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		exit_with_error("[file:%s line:%s] CUDA Error:%s", 
                        file, line, cudaGetErrorString(errorCode) );
	}
    errorCode = cudaThreadSynchronize();
    if (errorCode != cudaSuccess) {
		exit_with_error("[file:%s line:%s] CUDA sync Error%s:", 
                        file, line, cudaGetErrorString(errorCode) );
    }
}
