
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include "Point.h"

bool loadFromPTSFile(std::string filename, unsigned long long * initialSize, Point ** cloud, Point ** cloudCuda);

int main()
{
	//PointCloudDecimator * decimator = new PointCloudDecimator();
	//decimator->LoadFromPTSFile("SONGA_BREEZE_L4.pts");
	unsigned long long initialSize;
	Point * cloud = NULL;
	Point * cloudCuda = NULL;
	loadFromPTSFile("SONGA_BREEZE_L4.pts", &initialSize, &cloud, &cloudCuda);
	return 0;
}

bool loadFromPTSFile(std::string filename, unsigned long long * initialSize, Point ** cloud, Point ** cloudCuda)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	Point * point;
	Point * cloudLocal;
	Point * cloudCudaLocal;
	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			*initialSize = std::stoull(size);
			if (NULL != (cloudLocal = (Point *)malloc(*initialSize * sizeof(Point))))
			{
				if (cudaSuccess == cudaMalloc(&cloudCuda, *initialSize * sizeof(Point)))
				{
					for (unsigned long long i = 0; getline(ifs, line) && i < 100000; i++)
					{
						point = new Point();
						point->Parse(line);
						cloudLocal[i] = *point;
					}
					ifs.close();
					*cloud = cloudLocal;
					if (cudaSuccess == cudaMemcpy(cloudCudaLocal, cloudLocal, *initialSize, cudaMemcpyHostToDevice))
					{
						*cloudCuda = cloudCudaLocal;
						return true;
					}
				}
			}
		}
		ifs.close();
	}
	return false;
}
