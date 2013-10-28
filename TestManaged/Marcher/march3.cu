#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "typedefs.h"
#include "device_launch_parameters.h"
#include "util.hcu"
#include "cutil_math.h"
#include "tables.h"
#include <thrust\device_ptr.h>
#include <thrust/scan.h>

#include <cub/device/device_scan.cuh>



using namespace std;

extern "C" {

	__constant__ uint d_edgeTable[EDGE_SIZE];
	__constant__ uint d_triTable[TRI_ROWS][TRI_COLS];
	__constant__ uint d_vertTable[TRI_ROWS];

	const uint MAX_TRIANGLES = 15;


	extern "C" void allocateTables() {
		cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
		cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
		cudaMemcpyToSymbol(d_vertTable, numVertsTable, sizeof(numVertsTable));
	}

	__device__ 
		float3 cornerValue(const uint3 co, const float3 minX, const float3 dx) {
			return make_float3(minX.x + co.x*dx.x, minX.y + co.y*dx.y, minX.z + co.z*dx.z); 
	}

	__device__
	float tangle(float x, float y, float z)
	{
		x *= 3.0f;
		y *= 3.0f;
		z *= 3.0f;
		return (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;
	}

	__device__ 
		float func(float3 co) {
			//return co.x*co.x + co.y*co.y + co.z*co.z;
			/*return pow(co.x + co.y + co.z, 2) + co.x*co.x + __sinf(co.y)*co.y;*/
			//return pow(co.x, 3) + pow(co.y,3) + pow(co.z, 3) - co.x - co.y - co.z;
			return tangle(co.x, co.y, co.z);
	}

	__device__ 
		void interpValues(float isoValue, const float v0, const float v1, float3 p0, float3 p1, float3& out) {
			float mu = (isoValue - v0) / (v1 - v0);
			out = lerp(p0, p1, mu);
	}

	__device__
		float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
	{
		float3 edge0 = *v1 - *v0;
		float3 edge1 = *v2 - *v0;
		// note - it's faster to perform normalization in vertex shader rather than here
		return cross(edge0, edge1);
	}

	__global__
		void countKernel(float isoValue, dim3 dims, float3 minX, float3 dx, uint* count, uint* active) {
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

#pragma unroll
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

			uint nVert = d_vertTable[cubeindex];
			count[idx] = nVert;
			active[idx] = nVert > 0;
	}

	extern "C" void launchClassify(float isoValue, dim3 dims, float3 minX, float3 dx, uint* count, uint* isActive, uint N) {
		int blockSize = 4*32;
		int nBlocks = N/blockSize + (N%blockSize != 0);
		countKernel <<< nBlocks, blockSize >>> (isoValue, dims, minX, dx, count, isActive);
	}


	__global__
		void fillTriangles(float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, float3* norm, uint* vOffset, uint* cIndex, uint N) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;
			if (idx >= N) return;

			uint cIdx = cIndex[idx];
			uint3 co = idx_to_co(cIdx, dims);


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
#pragma unroll
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


			uint offset = vOffset[cIdx];

#pragma unroll
			for (uint i = 0; i < MAX_TRIANGLES; i+=3) {
				uint edge = d_triTable[cubeindex][i];
				uint edge1 = d_triTable[cubeindex][i+1];
				uint edge2 = d_triTable[cubeindex][i+2];
				if (edge == 255) break;

				float3 *v[3];
				v[0] = &vertList[edge];
				v[1] = &vertList[edge1];
				v[2] = &vertList[edge2];

				float3 n = -calcNormal(v[0], v[1], v[2]);

				norm[offset + i] = n;
				norm[offset + i + 1] = n;
				norm[offset + i + 2] = n;

				out[offset + i]     = vertList[edge];
				out[offset + i + 1] = vertList[edge1];
				out[offset + i + 2] = vertList[edge2];
			}
	}

	extern "C" void launchFill(float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, float3* norm, uint* vertexPrefix, uint* cubeIndex, uint N) {
		int blockSize = 8*32;
		int nBlocks = N/blockSize + (N%blockSize != 0);

		// void fillTriangles(float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, uint* vOffset, uint* cIndex, uint N)
		fillTriangles <<< nBlocks, blockSize >>> (isoValue, dims, minX, dx, out, norm, vertexPrefix, cubeIndex, N); 
	}


	extern "C" void exclusiveScan(uint* in, uint* out, uint N) {
		using namespace cub;
		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, N);
		// Allocate temporary storage for exclusive prefix sum
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, N);

		cudaFree(d_temp_storage); // Delete tmp
	}

	extern "C" uint retrieve(uint* array, uint element) {
		uint result;
		cudaMemcpy(&result, (array + element), sizeof(uint), cudaMemcpyDeviceToHost);
		return result;
	}



	__global__
		void compact(uint* isActive, uint* activeScan, uint* activeCompact, uint N) {
			uint idx = blockIdx.x*blockDim.x + threadIdx.x;
			if (isActive[idx]  && idx < N) {
				activeCompact[activeScan[idx]] = idx;
			}
	}

	extern "C" void launchCompact(uint* isActive, uint* activeScan, uint* activeCompact, uint N) {
		int blockSize = 4*32;
		int nBlocks = N/blockSize + (N%blockSize != 0);

		compact <<< nBlocks, blockSize >>> (isActive, activeScan, activeCompact, N);
	}


	int main2() {
		allocateTables();

		uint3 dims = make_uint3(5, 5, 5);
		float3 min = make_float3(1, 1, 1)*-1.1f;
		float3 dx = make_float3(2.0f / dims.x, 2.0f / dims.y, 2.0f / dims.z);

		const uint N = prod(dims);
		uint* d_count, *d_active, *d_activeScan, *d_compact;
		float3* d_pos;

		cudaMalloc((void **) &d_count, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_active, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_activeScan, (N+1)*sizeof(uint));
		cudaMalloc((void **) &d_pos, N*MAX_TRIANGLES*sizeof(float3));
		
		launchClassify(1, dims, min, dx, d_compact, d_active, N);
		
		exclusiveScan(d_active, d_activeScan, N+1);
		uint nVoxel = retrieve(d_activeScan, N);

		if (nVoxel == 0) {
			cout << "No voxels" << endl;
			return 0;
		}

		exclusiveScan(d_count, d_count, N+1);
		uint nVertex = retrieve(d_count, N);
		
		cudaMalloc((void **) &d_compact, nVoxel*sizeof(uint));
		launchCompact(d_active, d_activeScan, d_compact, N);
		
		float3* d_pos, *d_norm;
		cudaMalloc((void **) &d_pos, nVertex*sizeof(float3));
		cudaMalloc((void **) &d_norm, nVertex*sizeof(float3));

		launchFill(1, dims, min, dx, d_pos, d_norm, d_count, d_compact, nVoxel);


		{
			float3* h_pos2 = new float3[nVertex];
			cudaMemcpy(h_pos2, d_pos, nVertex * sizeof(float3), cudaMemcpyDeviceToHost);

			float3* h_norm = new float3[nVertex];
			cudaMemcpy(h_norm, d_norm, nVertex * sizeof(float3), cudaMemcpyDeviceToHost);

			for (int i = 0; i < nVertex; i++) {
				float3 f = h_pos2[i];
				cout << f.x << "    " << f.y << "     " << f.z << "     " << f.x*f.x + f.y*f.y + f.z*f.z << endl;
			}

			cout << "Norms " << endl << endl << endl << endl << endl << endl << endl << endl;

			for (int i = 0; i < nVertex; i++) {
				float3 f = h_norm[i];
				cout << f.x << "    " << f.y << "     " << f.z << "     " << f.x*f.x + f.y*f.y + f.z*f.z << endl;
			}
		}


		return 1;
	}
}