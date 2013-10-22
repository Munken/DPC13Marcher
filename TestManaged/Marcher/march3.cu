#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "typedefs.h"
#include "device_launch_parameters.h"
#include "util.hcu"
#include "cutil_math.h"
#include "tables.h"

using namespace std;

extern "C" {

__constant__ uint d_edgeTable[EDGE_SIZE];
__constant__ uint d_triTable[TRI_ROWS][TRI_COLS];

const uint MAX_TRIANGLES = 15;


void allocateTables() {
	cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
	cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
}

__device__ 
	float3 cornerValue(const uint3 co, const float3 minX, const float3 dx) {
		return make_float3(minX.x + co.x*dx.x, minX.y + co.y*dx.y, minX.z + co.z*dx.z); 
}

__device__ 
	float func(float3 co) {
		return co.x*co.x + co.y*co.y + co.z*co.z - 1;
}

__device__ 
	void interpValues(float isoValue, const float v0, const float v1, float3 p0, float3 p1, float3& out) {
		float mu = (isoValue - v0) / (v1 - v0);
		out = lerp(p0, p1, mu);
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

			interpValues(isoValue,value[0],value[1],corners[0],corners[1], vertList[0]);
			interpValues(isoValue,value[1],value[2],corners[1],corners[2], vertList[1]);
			interpValues(isoValue,value[2],value[3],corners[2],corners[3], vertList[2]);
			interpValues(isoValue,value[3],value[0],corners[3],corners[0], vertList[3]);
			interpValues(isoValue,value[4],value[5],corners[4],corners[5], vertList[4]);
			interpValues(isoValue,value[5],value[6],corners[5],corners[6], vertList[5]);
			interpValues(isoValue,value[6],value[7],corners[6],corners[7], vertList[6]);
			interpValues(isoValue,value[7],value[4],corners[7],corners[4], vertList[7]);
			interpValues(isoValue,value[0],value[4],corners[0],corners[4], vertList[8]);
			interpValues(isoValue,value[1],value[5],corners[1],corners[5], vertList[9]);
			interpValues(isoValue,value[2],value[6],corners[2],corners[6], vertList[10]);
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


int main() {
	allocateTables();

	uint3 dims = make_uint3(20, 20, 20);
	float3 min = make_float3(1, 1, 1)*-3;
	float3 dx = make_float3(0.2f, 0.2f, 0.2f);

	const uint N = prod(dims);
	uint* d_count;
	float3* d_pos;

	cudaMalloc((void **) &d_count, N*sizeof(uint));
	cudaMalloc((void **) &d_pos, N*MAX_TRIANGLES*sizeof(float3));


	
	simpleKernel <<< N/200, 200 >>> (0, dims, min, dx, d_pos, d_count);

	uint* h_count = new uint[N];
	float3* h_pos = new float3[N*MAX_TRIANGLES];
	cudaMemcpy(h_count, d_count, N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pos, d_pos, N * MAX_TRIANGLES * sizeof(float3), cudaMemcpyDeviceToHost);


	for (uint i = 0; i < N; i++) {
		uint h = h_count[i];
		if (!h) continue;

		cout << h << endl;
		/*for (uint j = 0; j < h; j++) {
			float3 f = h_pos[i*MAX_TRIANGLES + j];
			cout << f.x << "    " << f.y << "     " << f.z << "     " << f.x*f.x + f.y*f.y + f.z*f.z << endl;
		}
		cout << endl;*/
	}
	return 1;
}
}