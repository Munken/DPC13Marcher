#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "typedefs.h"
#include "device_launch_parameters.h"

#include <thrust\scan.h>
#include <thrust\device_ptr.h>

#include <cub/device/device_scan.cuh>

#include "util.hcu"
#include "cutil_math.h"
#include "tables.h"
#include "GPUTimer.h"
#include "cuda_utilities.h"

using namespace std;

extern "C" {

	__constant__ uint d_edgeTable[EDGE_SIZE];
	__constant__ uint d_triTable[TRI_ROWS][TRI_COLS];
	__constant__ uint d_countTable[TRI_ROWS];

	const uint MAX_TRIANGLES = 15;


	void allocateTables() {
		cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
		cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
		cudaMemcpyToSymbol(d_countTable, numVertsTable, sizeof(numVertsTable));
	}

	__device__ 
		inline float3 cornerValue(const uint3 co, const float3 minX, const float3 dx) {
			return make_float3(minX.x + co.x*dx.x, minX.y + co.y*dx.y, minX.z + co.z*dx.z); 
	}

	__device__ 
		inline float func(float3 co) {
			return co.x*co.x + co.y*co.y + co.z*co.z - 1;
	}

	__device__ 
		inline void interpValues(float isoValue, const float v0, const float v1, float3 p0, float3 p1, float3& out) {
			float mu = (isoValue - v0) / (v1 - v0);
			out = lerp(p0, p1, mu);
	}

	__device__
		inline uint getEdge(uint i) {
			return edgeTable[i];
	}

	__global__ 
		void countKernel(float isoValue, dim3 dims, float3 minX, float3 dx, uint* count, uint* isOccupied, uint N) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;

			if (idx >= N) return;


			uint3 co = idx_to_co(idx, dims);

			float3 corners[8];
			corners[0] = cornerValue(co, minX, dx);
			corners[1] = corners[0] + make_float3(dx.x, 0,    0);
			corners[2] = corners[0] + make_float3(dx.x, dx.y, 0);
			corners[3] = corners[0] + make_float3(0,    dx.y, 0);

			corners[4] = corners[0] + make_float3(0,    0,    dx.z);
			corners[5] = corners[0] + make_float3(dx.x, 0,    dx.z);
			corners[6] = corners[0] + make_float3(dx.x, dx.y, dx.z);
			corners[7] = corners[0] + make_float3(0,    dx.y, dx.z);

//			float value[8];
//#pragma unroll 8
//			for (int i = 0; i < 8; i++) {
//				value[i] = func(corners[i]);
//			}

			uint cubeindex;
			cubeindex =  uint(func(corners[0]) < isoValue); 
			cubeindex += uint(func(corners[1]) < isoValue)*2; 
			cubeindex += uint(func(corners[2]) < isoValue)*4; 
			cubeindex += uint(func(corners[3]) < isoValue)*8; 
			cubeindex += uint(func(corners[4]) < isoValue)*16; 
			cubeindex += uint(func(corners[5]) < isoValue)*32; 
			cubeindex += uint(func(corners[6]) < isoValue)*64; 
			cubeindex += uint(func(corners[7]) < isoValue)*128;

			uint nVertices = d_countTable[cubeindex];
			count[idx] = nVertices;
			isOccupied[idx] = nVertices > 0;
	}

	__global__
		void compact(uint* isOccupied, uint* occupiedScan, uint* occupiedCompact) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;
			if (isOccupied[idx]) {
				occupiedCompact[occupiedScan[idx]] = idx;
			}
	}

	__global__
		void fillTriangles(float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, uint* vertexPrefix, uint N) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;

			if (idx >= N) return;

			uint3 co = idx_to_co(idx, dims);

			float3 corners[8];
			corners[0] = cornerValue(co, minX, dx);
			corners[1] = corners[0] + make_float3(dx.x, 0,    0);
			corners[2] = corners[0] + make_float3(dx.x, dx.y, 0);
			corners[3] = corners[0] + make_float3(0,    dx.y, 0);

			corners[4] = corners[0] + make_float3(0,    0,    dx.z);
			corners[5] = corners[0] + make_float3(dx.x, 0,    dx.z);
			corners[6] = corners[0] + make_float3(dx.x, dx.y, dx.z);
			corners[7] = corners[0] + make_float3(0,    dx.y, dx.z);

			float value[8];

#pragma unroll 8
			for (int i = 0; i < 8; i++) {
				value[i] = func(corners[i]);
			}

			uint cubeindex;
			cubeindex =  uint(value[0] < isoValue); 
			cubeindex += uint(value[1] < isoValue)*2; 
			cubeindex += uint(value[2] < isoValue)*4; 
			cubeindex += uint(value[3] < isoValue)*8; 
			cubeindex += uint(value[4] < isoValue)*16; 
			cubeindex += uint(value[5] < isoValue)*32; 
			cubeindex += uint(value[6] < isoValue)*64; 
			cubeindex += uint(value[7] < isoValue)*128;



			float3 vertList[12];

			if (getEdge(cubeindex) & 1)
				interpValues(isoValue,value[0],value[1],corners[0],corners[1], vertList[0]);
			if (getEdge(cubeindex) & 2)
				interpValues(isoValue,value[1],value[2],corners[1],corners[2], vertList[1]);
			if (getEdge(cubeindex) & 4)
				interpValues(isoValue,value[2],value[3],corners[2],corners[3], vertList[2]);
			if (getEdge(cubeindex) & 8)
				interpValues(isoValue,value[3],value[0],corners[3],corners[0], vertList[3]);
			if (getEdge(cubeindex) & 16)
				interpValues(isoValue,value[4],value[5],corners[4],corners[5], vertList[4]);
			if (getEdge(cubeindex) & 32)
				interpValues(isoValue,value[5],value[6],corners[5],corners[6], vertList[5]);
			if (getEdge(cubeindex) & 64)
				interpValues(isoValue,value[6],value[7],corners[6],corners[7], vertList[6]);
			if (getEdge(cubeindex) & 128)
				interpValues(isoValue,value[7],value[4],corners[7],corners[4], vertList[7]);
			if (getEdge(cubeindex) & 256)
				interpValues(isoValue,value[0],value[4],corners[0],corners[4], vertList[8]);
			if (getEdge(cubeindex) & 512)
				interpValues(isoValue,value[1],value[5],corners[1],corners[5], vertList[9]);
			if (getEdge(cubeindex) & 1024)
				interpValues(isoValue,value[2],value[6],corners[2],corners[6], vertList[10]);
			if (getEdge(cubeindex) & 2048)
				interpValues(isoValue,value[3],value[7],corners[3],corners[7], vertList[11]);

			
			const uint offset = vertexPrefix[idx];

			for (uint i = 0; i < MAX_TRIANGLES; i++) {
				uint edge = d_triTable[cubeindex][i];
				if (edge == 255) break;

				out[offset + i] = vertList[edge];
			}
	}

	void exclusiveScan(uint* in, uint* out, uint N) {
		/*thrust::exclusive_scan(thrust::device_ptr<unsigned int>(in),
			thrust::device_ptr<unsigned int>(in + N),
			thrust::device_ptr<unsigned int>(out));*/

		using namespace cub;
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, N);
		// Allocate temporary storage for exclusive prefix sum
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, N);

		cudaFree(d_temp_storage);
	}

	uint retrieve(uint* array, uint element) {
		uint result;
		cudaMemcpy(&result, (array + element), sizeof(uint), cudaMemcpyDeviceToHost);
		return result;
	}

	int main() {
		using namespace Gadgetron;
		GPUTimer* t;

		t = new GPUTimer("Const alloc");
		allocateTables();
		delete t;

		
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(countKernel, cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(fillTriangles, cudaFuncCachePreferL1);
		int n = 100;
		uint3 dims = make_uint3(1, 1, 1) * n;
		float3 min = make_float3(1, 1, 1)*-1.2f;
		float3 dx = make_float3(1, 1, 1)*0.02f;

		const uint N = prod(dims);
		uint* d_count;
		uint *d_occupied, *d_occupiedCompact, *d_occupiedScan;
		float3* d_pos;


		//t = new GPUTimer("Malloc");
		cudaMalloc((void **) &d_count, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_occupied, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_occupiedScan, (N+1)*sizeof(uint));
		//delete t;

		//t = new GPUTimer("Running kernel");
		cudaThreadSynchronize();
		t = new GPUTimer("Total");

		{
		int blockSize = 12*32;
		int nBlocks = N/blockSize + (N%blockSize != 0);
		countKernel <<< nBlocks, blockSize >>> (0, dims, min, dx, d_count, d_occupied, N);
		}
		//cudaThreadSynchronize();
		//delete t;
		//CHECK_FOR_CUDA_ERROR();

		//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		
		//t = new GPUTimer("Scan occupied");
		exclusiveScan(d_occupied, d_occupiedScan, N+1);
		//delete t;

		//t = new GPUTimer("Transfer last occupied element");
		uint nVoxel = retrieve(d_occupiedScan, N);
		//cout << nVoxel << endl;
		//delete t;

		//t = new GPUTimer("Malloc compact");
		cudaMalloc((void **) &d_occupiedCompact, nVoxel*sizeof(uint));
		//delete t;

		//t = new GPUTimer("Compact");
		{
		int blockSize = 1*n;
		int nBlocks = N/blockSize + (N%blockSize != 0);
		compact <<< nBlocks, blockSize >>> (d_occupied, d_occupiedScan, d_occupiedCompact);
		}
		//delete t;

		//t = new GPUTimer("Scan count");
		exclusiveScan(d_count, d_count, N+1);
		//delete t;

		//t = new GPUTimer("Transfer last scan element");
		uint nVertex = retrieve(d_count, N);
		//cout << nVertex << endl;
		//delete t;
		//CHECK_FOR_CUDA_ERROR();

		//t = new GPUTimer("Alloc vertex array");
		cudaMalloc((void **) &d_pos, nVertex*sizeof(float3));
		//delete t;
		//CHECK_FOR_CUDA_ERROR();

		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		//t = new GPUTimer("Gen triangles");
		{
		int blockSize = 5*32;
		int nBlocks = nVoxel/blockSize + (nVoxel%blockSize != 0);
		fillTriangles <<< nBlocks, blockSize >>> (0, dims, min, dx, d_pos, d_count, nVoxel);
		}
		//delete t;
		//CHECK_FOR_CUDA_ERROR();


		float3* h_pos = new float3[nVertex];
		//t = new GPUTimer("Memcpy");
		cudaMemcpy(h_pos, d_pos, nVertex * sizeof(float3), cudaMemcpyDeviceToHost);
		delete t;
		//CHECK_FOR_CUDA_ERROR();

		return 0;
		
	}
}
