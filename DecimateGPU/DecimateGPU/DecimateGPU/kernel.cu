
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>

typedef unsigned long long ull;
typedef short int si;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void makeCubesKernel(const si d, bool * removedGpu, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
{
	ull i = blockIdx.x*blockDim.x + threadIdx.x;
	xGpu[i] = pointsGpu[i].x % d;
	yGpu[i] = pointsGpu[i].y % d;
	zGpu[i] = pointsGpu[i].z % d;
	bxGpu[i] = pointsGpu[i].x / d;
	byGpu[i] = pointsGpu[i].y / d;
	bzGpu[i] = pointsGpu[i].z / d;

}

__global__ void filterCubesKernel(const si d, const ull arrLength, bool * removedGpu, si * xGpu, si * yGpu, si * zGpu, si * bxGpu, si * byGpu, si * bzGpu, Point * pointsGpu)
{
	ull i = blockIdx.x*blockDim.x + threadIdx.x;
	ull j = i + 1;
	for (; j < arrLength && bxGpu[i] != bxGpu[j] && byGpu[i] != byGpu[j] && bzGpu[i] != bzGpu[j]; j++);
	if (j < arrLength)
	{
		removedGpu[j-1] = true;
	}
}

int main()
{
	si d = 7;

	FILE * cloudFileBin = fopen("S:\SONGA_BREEZE_L4.bin", "rb");

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
	ull buffSize = vramSizeInBytes - (vramSizeInBytes % sizeOfPoint);
	Point * cloudBuffer = (Point *)malloc(buffSize);

	ull pointsCount = fread(cloudBuffer, sizeOfPoint, buffCount, cloudFileBin);

	cudaError_t cudaStatus;

	bool * removed = (bool *)calloc(pointsCount, sizeof(bool));
	si * x = (si *)malloc(pointsCount * sizeof(si));
	si * y = (si *)malloc(pointsCount * sizeof(si));
	si * z = (si *)malloc(pointsCount * sizeof(si));
	si * bx = (si *)malloc(pointsCount * sizeof(si));
	si * by = (si *)malloc(pointsCount * sizeof(si));
	si * bz = (si *)malloc(pointsCount * sizeof(si));

    bool * removedGpu = false;
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

	int blockSize;
	int minGridSize;
	int gridSize;

	cudaStatus = cudaMalloc((void**)&removedGpu, pointsCount * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&xGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&yGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&zGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&bxGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&byGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&bzGpu, pointsCount * sizeof(si));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&pointsGpu, pointsCount * sizeOfPoint);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(pointsGpu, cloudBuffer, pointsCount * sizeOfPoint, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	ull blocksCount = 1024;
	makeCubesKernel <<<pointsCount/blocksCount, blocksCount >>> (d, removedGpu, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "makeCubesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(x, xGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(bx, bxGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// TEST

	filterCubesKernel <<<pointsCount / blocksCount, blocksCount >>> (d, pointsCount, removedGpu, xGpu, yGpu, zGpu, bxGpu, byGpu, bzGpu, pointsGpu);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "makeCubesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(removed, removedGpu, pointsCount * sizeof(si), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
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
