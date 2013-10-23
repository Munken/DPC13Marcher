#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "typedefs.h"
#include "device_launch_parameters.h"

#include <thrust\scan.h>
#include <thrust\device_ptr.h>

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

	__global__ 
		void countKernel(float isoValue, dim3 dims, float3 minX, float3 dx, uint* count, uint* isOccupied) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;

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
		void simpleKernel(float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, uint* count) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;

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

			if (d_edgeTable[cubeindex] & 1)
				interpValues(isoValue,value[0],value[1],corners[0],corners[1], vertList[0]);
			if (d_edgeTable[cubeindex] & 2)
				interpValues(isoValue,value[1],value[2],corners[1],corners[2], vertList[1]);
			if (d_edgeTable[cubeindex] & 4)
				interpValues(isoValue,value[2],value[3],corners[2],corners[3], vertList[2]);
			if (d_edgeTable[cubeindex] & 8)
				interpValues(isoValue,value[3],value[0],corners[3],corners[0], vertList[3]);
			if (d_edgeTable[cubeindex] & 16)
				interpValues(isoValue,value[4],value[5],corners[4],corners[5], vertList[4]);
			if (d_edgeTable[cubeindex] & 32)
				interpValues(isoValue,value[5],value[6],corners[5],corners[6], vertList[5]);
			if (d_edgeTable[cubeindex] & 64)
				interpValues(isoValue,value[6],value[7],corners[6],corners[7], vertList[6]);
			if (d_edgeTable[cubeindex] & 128)
				interpValues(isoValue,value[7],value[4],corners[7],corners[4], vertList[7]);
			if (d_edgeTable[cubeindex] & 256)
				interpValues(isoValue,value[0],value[4],corners[0],corners[4], vertList[8]);
			if (d_edgeTable[cubeindex] & 512)
				interpValues(isoValue,value[1],value[5],corners[1],corners[5], vertList[9]);
			if (d_edgeTable[cubeindex] & 1024)
				interpValues(isoValue,value[2],value[6],corners[2],corners[6], vertList[10]);
			if (d_edgeTable[cubeindex] & 2048)
				interpValues(isoValue,value[3],value[7],corners[3],corners[7], vertList[11]);

			uint i = 0;
			uint offset = idx*MAX_TRIANGLES;
			count[idx] = 25;

			for (; i < MAX_TRIANGLES; i++) {
				uint edge = d_triTable[cubeindex][i];
				count[idx] = i;
				if (edge == 255) break;

				out[offset + i] = vertList[edge];
			}

			count[idx] = i;
	}

	void exclusiveScan(uint* in, uint* out, uint N) {
		thrust::exclusive_scan(thrust::device_ptr<unsigned int>(in),
			thrust::device_ptr<unsigned int>(in + N),
			thrust::device_ptr<unsigned int>(out));
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

		int n = 70;
		uint3 dims = make_uint3(1, 1, 1) * n;
		float3 min = make_float3(1, 1, 1)*-1.2f;
		float3 dx = make_float3(0.2f, 0.2f, 0.2f);

		const uint N = prod(dims);
		uint* d_count;
		uint *d_occupied, *d_occupiedCompact, *d_occupiedScan;
		float3* d_pos;

		t = new GPUTimer("Malloc");
		cudaMalloc((void **) &d_count, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_occupied, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_occupiedScan, (N+1)*sizeof(uint));
		delete t;

		t = new GPUTimer("Running kernel");
		//simpleKernel <<< N/n, n >>> (0, dims, min, dx, d_pos, d_count);
		countKernel <<< N/n, n >>> (0, dims, min, dx, d_count, d_occupied);
		delete t;
		CHECK_FOR_CUDA_ERROR();

		
		t = new GPUTimer("Scan occupied");
		exclusiveScan(d_occupied, d_occupiedScan, N+1);
		delete t;

		t = new GPUTimer("Transfer last occupied element");
		uint nVoxel = retrieve(d_occupiedScan, N);
		cout << nVoxel << endl;
		delete t;

		t = new GPUTimer("Malloc compact");
		cudaMalloc((void **) &d_occupiedCompact, nVoxel*sizeof(uint));
		delete t;

		t = new GPUTimer("Compact");
		compact <<< N/n, n >>> (d_occupied, d_occupiedScan, d_occupiedCompact);
		delete t;

		t = new GPUTimer("Scan count");
		exclusiveScan(d_occupied, d_occupiedScan, N+1);
		delete t;

		t = new GPUTimer("Transfer last scan element");
		uint nVertex = retrieve(d_count, N);
		cout << nVertex << endl;
		delete t;
		CHECK_FOR_CUDA_ERROR();


		return 0;
		uint* h_count = new uint[N];
		float3* h_pos = new float3[N*MAX_TRIANGLES];

		t = new GPUTimer("Memcpy");
		cudaMemcpy(h_count, d_count, N * sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_pos, d_pos, N * MAX_TRIANGLES * sizeof(float3), cudaMemcpyDeviceToHost);
		delete t;
		CHECK_FOR_CUDA_ERROR();

		uint count = 0;
		for (uint i = 0; i < N; i++) {
			uint h = h_count[i];
			if (!h) continue;
			count+=h;

			//cout << h << endl;
			/*for (uint j = 0; j < h; j++) {
			float3 f = h_pos[i*MAX_TRIANGLES + j];
			cout << f.x << "    " << f.y << "     " << f.z << "     " << f.x*f.x + f.y*f.y + f.z*f.z << endl;
			}
			cout << endl;*/
		}
		cout << count << endl;
		return 1;
	}
}
