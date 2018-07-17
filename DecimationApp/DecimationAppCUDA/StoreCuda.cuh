#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <list>

#include "Point.h"
#include "Store.h"

namespace StoreCuda
{

	struct StoreCuda
	{
		Store::Store * store;
		ull xDecMap, yDecMap, zDecMap;
		fraction_list xFracMap, yFracMap, zFracMap;
		fraction_list xFracList, yFracList, zFracList;
		double theD;
	};

	StoreCuda * init(Store::Store * store);
	
	void copyDataToGpu(StoreCuda * storeCuda);

	void copyDimensionToGpu(StoreCuda * storeCuda, decimal_record_list * dimension, ull * decListMap, fraction_list * fracListMap, fraction_list * fracListCuda);

	void decimate(StoreCuda * storeCuda);

}