
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"

#include "TxtToBinConverter.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

#define PRIME_1 157 // 73856093
#define PRIME_2 183 // 19349663
#define PRIME_3 386 // 83492791

#define MAKE_3D_HASH(x,y,z) \
((x) * PRIME_1) ^ ((y) * PRIME_2) ^ ((z) * PRIME_3);

#define FILTER_BLOCKS_COUNT_Y 10000
#define FILTER_BLOCKS_CHUNK_SIZE 1000000

#define FILTER_HIT_SIZE 100000

#define CUDA_MEMCPY_FAIL_CHECK(cudaStatus) \
if (cudaStatus != cudaSuccess) { \
	fprintf(stderr, "cudaMemcpy failed!"); \
	goto Error; \
}
#define CUDA_MALLOC_FAIL_CHECK(cudaStatus) \
if (cudaStatus != cudaSuccess) { \
	fprintf(stderr, "cudaMalloc failed!"); \
	goto Error; \
}

typedef unsigned long long ull;
typedef short int si;

__global__ void makeCubesKernel(const si d, const ull arrSize, bool * removedGpu, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
{
	ull i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < arrSize)
	{
		xGpu[i] = pointsGpu[i].x % d;
		yGpu[i] = pointsGpu[i].y % d;
		zGpu[i] = pointsGpu[i].z % d;
		bxGpu[i] = pointsGpu[i].x / d;
		byGpu[i] = pointsGpu[i].y / d;
		bzGpu[i] = pointsGpu[i].z / d;
	}

}

__global__ void filterCubesKernel(const si d, const ull arrLength, bool * removedGpu, bool * siblingFound, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
{
	ull i, j, jLimit;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < arrLength && !siblingFound[i])
	{
		jLimit = (blockIdx.y + 1) * FILTER_BLOCKS_CHUNK_SIZE;
		//ull j = i + (blockIdx.y * FILTER_BLOCKS_COUNT_Y);
		//ull searchLimit = j + FILTER_BLOCKS_COUNT_Y;
		for (
			j = i + 1;
			j < arrLength
			&& j < jLimit
			//&& j < 100000
			/*&& !(bxGpu[i] == bxGpu[j]
				&& byGpu[i] == byGpu[j]
				&& bzGpu[i] == bzGpu[j])*/;
			j++)
		{
			if (bxGpu[i] == bzGpu[j]
				&& byGpu[i] == byGpu[j]
				&& bzGpu[i] == bzGpu[j])
			{
				break;
			}
		}
		if (j < arrLength 
			&& bxGpu[i] == bzGpu[j]
			&& byGpu[i] == byGpu[j]
			&& bzGpu[i] == bzGpu[j])
		{
			removedGpu[j] = true;
			siblingFound[i] = true;
		}
	}
}

__global__ void filterCubesCmpKernel(const si d, const ull baseIndex, si * xGpuTmp, si * yGpuTmp, si * zGpuTmp, si * bxGpuTmp, si * byGpuTmp, si * bzGpuTmp, const ull arrLength, bool * removedGpu, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
{
	ull i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x + baseIndex;
	if (i < arrLength && !removedGpu[i])
	{
		for (j = 0; j < FILTER_HIT_SIZE; j++)
		{
			if (zGpuTmp[j] == zGpu[i]
				&& yGpuTmp[j] == yGpu[i]
				&& zGpuTmp[j] == zGpu[i]
				&& bxGpuTmp[j] == bxGpu[i]
				&& byGpuTmp[j] == byGpu[i]
				&& bzGpuTmp[j] == bzGpu[i])
			{
				removedGpu[i] = true;
			}
		}
	}
}

__global__ void mapCubesKernel(const ull arrLen, const ull mapLen, const ull dimOffset, si * bxGpu, si * byGpu, si * bzGpu, bool * bMapIsInGpu, si * bMapGpu)
{
	ull i = blockIdx.x*blockDim.x + threadIdx.x;
	ull hash;
	if (i < arrLen)
	{
		hash = MAKE_3D_HASH(bxGpu[i] + dimOffset, byGpu[i] + dimOffset, bzGpu[i] + dimOffset);
		bMapIsInGpu[hash] = true;
		bMapGpu[hash] = i;
	}
}

int main()
{

	//auto converter = new TxtToBinConverter();
	//converter->SetTextFileName("D:\\decimate\\SONGA_BREEZE_L4.pts");
	//converter->SetBinFileName("K:\\SONGA_BREEZE_L4.bin");
	//converter->Convert();
	//delete converter;

	si d = 7;

	FILE * cloudFileBin = fopen("K:\SONGA_BREEZE_L4.bin", "rb");

	if (cloudFileBin == NULL)
	{
		return 1;
	}
	
	const ull sizeOfPoint = sizeof(struct Point);

	fseek(cloudFileBin, 0L, SEEK_END);
	ull binCloudFileSize = ftell(cloudFileBin);
	ull cloudCount = binCloudFileSize / sizeOfPoint;
	rewind(cloudFileBin);

	ull vramSizeInBytes = 1 * 1024 * 1024 * 1024;
	//vramSizeInBytes = 1920 * sizeOfPoint;
	ull buffCount = vramSizeInBytes / sizeOfPoint;
	buffCount = 1000000;
	ull buffSize = vramSizeInBytes - (vramSizeInBytes % sizeOfPoint);
	Point * cloudBuffer = (Point *)malloc(buffSize);

	ull pointsCount;
	cudaError_t cudaStatus;

	bool * removed = (bool *)calloc(cloudCount, sizeof(bool));
	si * x = (si *)malloc(cloudCount * sizeof(si));
	si * y = (si *)malloc(cloudCount * sizeof(si));
	si * z = (si *)malloc(cloudCount * sizeof(si));
	si * bx = (si *)malloc(cloudCount * sizeof(si));
	si * by = (si *)malloc(cloudCount * sizeof(si));
	si * bz = (si *)malloc(cloudCount * sizeof(si));

	bool * removedGpu = false;
	bool * siblingFound = false;
	si * xGpu = 0;
	si * yGpu = 0;
	si * zGpu = 0;
	si * bxGpu = 0;
	si * byGpu = 0;
	si * bzGpu = 0;

	Point * pointsGpu = NULL;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int maxBlockSize;
	int minGridSize;
	int gridSize;

	cudaStatus = cudaMalloc((void**)&removedGpu, buffCount * sizeof(bool));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&siblingFound, buffCount * sizeof(bool));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&xGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&yGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&zGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&bxGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&byGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&bzGpu, buffCount * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&pointsGpu, buffCount * sizeOfPoint);
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	ull blockSize = 1024; // 1024

#pragma region IndexingPrepRegion

	long long int bMin, bMax;
	ull mapLen;
	si * bMap;
	si * bMapGpu;

	bool * bIsInMap = false;
	bool * bIsInMapGpu = false;

	bMin = -100000;
	bMax = 100000;

	mapLen = abs(bMin) + abs(bMax) + 1;

	mapLen = MAKE_3D_HASH(mapLen, mapLen, mapLen);

	bIsInMap = (bool *)calloc(mapLen, sizeof(bool));
	bMap = (si *)malloc(mapLen * sizeof(ull));
	cudaStatus = cudaMalloc((void**)&bIsInMapGpu, mapLen * sizeof(bool));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&bMapGpu, mapLen * sizeof(ull));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);

#pragma endregion

	for (ull i = 0; (pointsCount = fread(cloudBuffer, sizeOfPoint, buffCount, cloudFileBin)) != 0; i += pointsCount)
	{
		cudaStatus = cudaMemcpy(pointsGpu, cloudBuffer, pointsCount * sizeOfPoint, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		makeCubesKernel << <pointsCount / blockSize + 1, blockSize >> > (d, pointsCount, removedGpu, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);

		mapCubesKernel << <pointsCount / blockSize + 1, blockSize >> > (pointsCount, mapLen, -bMin, bxGpu, byGpu, bzGpu, bIsInMapGpu, bMapGpu);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "makeCubesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching makeCubesKernel!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = cudaMemcpy(&x[i], xGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		cudaStatus = cudaMemcpy(&y[i], yGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		cudaStatus = cudaMemcpy(&z[i], zGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		cudaStatus = cudaMemcpy(&bx[i], bxGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		cudaStatus = cudaMemcpy(&by[i], byGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		cudaStatus = cudaMemcpy(&bz[i], bzGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

	}

	cudaStatus = cudaMemcpy(bIsInMap, bIsInMapGpu, mapLen * sizeof(bool), cudaMemcpyDeviceToHost);
	CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMemcpy(bMap, bMapGpu, mapLen * sizeof(ull), cudaMemcpyDeviceToHost);
	CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

	ull count = 0; // number of unique boxes
	for (ull i = 0; i < mapLen; i++)
	{
		if (bIsInMap[i]) count++;
	}

	return 0;

Error:
	return cudaStatus;
}