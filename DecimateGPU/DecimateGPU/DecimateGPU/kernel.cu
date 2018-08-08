
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"

#include "TxtToBinConverter.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

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

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

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

//__global__ void filterCubesCmpKernel(const si d, const ull baseIndex, const si x, const si y, const si z, const si bx, const si by, const si bz, const ull arrLength, bool * removedGpu, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
//{
//	ull i;
//	i = blockIdx.x*blockDim.x + threadIdx.x + baseIndex;
//	if (i < arrLength 
//		&& !removedGpu[i]
//		&& z == zGpu[i]
//		&& y == yGpu[i]
//		&& z == zGpu[i]
//		&& bx == bxGpu[i]
//		&& by == byGpu[i]
//		&& bz == bzGpu[i])
//	{
//		removedGpu[i] = true;
//	}
//}

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

	for (ull i = 0; (pointsCount = fread(cloudBuffer, sizeOfPoint, buffCount, cloudFileBin)) != 0; i += pointsCount)
	{
		cudaStatus = cudaMemcpy(pointsGpu, cloudBuffer, pointsCount * sizeOfPoint, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		makeCubesKernel << <pointsCount / blockSize + 1, blockSize >> > (d, pointsCount, removedGpu, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);

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

	// TEST

	dim3 gridDim(pointsCount / blockSize);
	//filterCubesKernel <<<gridDim, blockSize >>> (d, pointsCount, removedGpu, siblingFound, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);
	
	si * xGpuTmp = 0;
	si * yGpuTmp = 0;
	si * zGpuTmp = 0;
	si * bxGpuTmp = 0;
	si * byGpuTmp = 0;
	si * bzGpuTmp = 0;

	cudaStatus = cudaMalloc((void**)&xGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&yGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&zGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&bxGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&byGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);
	cudaStatus = cudaMalloc((void**)&bzGpuTmp, FILTER_HIT_SIZE * sizeof(si));
	CUDA_MALLOC_FAIL_CHECK(cudaStatus);

	si * xTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));
	si * yTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));
	si * zTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));
	si * bxTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));
	si * byTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));
	si * bzTmp = (si *)malloc(FILTER_HIT_SIZE * sizeof(si));

	for (ull i = 0; i < pointsCount; i+=FILTER_HIT_SIZE)
	{
		for (int j = 0; j < FILTER_HIT_SIZE; j++)
		{
			xTmp[j] = x[i + j];
			yTmp[j] = y[i + j];
			zTmp[j] = z[i + j];
			bxTmp[j] = bx[i + j];
			byTmp[j] = by[i + j];
			bzTmp[j] = bz[i + j];
		}
		cudaStatus = cudaMemcpy(xTmp, xGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
		cudaStatus = cudaMemcpy(yTmp, yGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
		cudaStatus = cudaMemcpy(zTmp, zGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
		cudaStatus = cudaMemcpy(bxTmp, bxGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
		cudaStatus = cudaMemcpy(byTmp, byGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);
		cudaStatus = cudaMemcpy(bzTmp, bzGpuTmp, FILTER_HIT_SIZE * sizeof(si), cudaMemcpyDeviceToHost);
		CUDA_MEMCPY_FAIL_CHECK(cudaStatus);

		ull gridSize;
		gridSize = (pointsCount - i) / blockSize;
		filterCubesCmpKernel << <gridSize, blockSize >> > (d, i, xGpuTmp, yGpuTmp, zGpuTmp, bxGpuTmp, byGpuTmp, bzGpuTmp, pointsCount, removedGpu, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);

		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "makeCubesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//	goto Error;
		//}

	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "makeCubesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching filterCubesKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(removed, removedGpu, pointsCount * sizeof(bool), cudaMemcpyDeviceToHost);
	CUDA_MEMCPY_FAIL_CHECK(cudaStatus);


	ull removedCount = 0;
	for (ull i = 0; i < pointsCount; i++)
	{
		if (removed[i])
		{
			removedCount++;
		}
	}

	// TEST END

	return 0;

Error:
	return cudaStatus;
}

int main2()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
