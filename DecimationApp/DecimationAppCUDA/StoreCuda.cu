#include "StoreCuda.cuh"

namespace StoreCuda
{
	StoreCuda * init(Store::Store * store)
	{
		StoreCuda * storeCuda = new StoreCuda();
		storeCuda->store = store;
		return storeCuda;
	}

	void copyDataToGpu(StoreCuda * storeCuda)
	{
		Store::Store * store = storeCuda->store;
		copyDimensionToGpu(storeCuda, &(store->x), &(storeCuda->xDecMap), &(storeCuda->xFracMap), &(storeCuda->xFracList));
		copyDimensionToGpu(storeCuda, &(store->y), &(storeCuda->yDecMap), &(storeCuda->yFracMap), &(storeCuda->yFracList));
		copyDimensionToGpu(storeCuda, &(store->z), &(storeCuda->zDecMap), &(storeCuda->zFracMap), &(storeCuda->zFracList));
	}

	void copyDimensionToGpu(StoreCuda * storeCuda, decimal_record_list * dimension, ull * decListMap, fraction_list * fracListMap, fraction_list * fracListCuda)
	{
		auto decCount = dimension->size();
		ull *decList;
		fraction_list *fracList;

		decList = (ull *)malloc(decCount * sizeof ull);
		fracList = (fraction_list *)malloc(decCount * sizeof fraction_list);

		auto fracCount = Store::calcFracListLength(storeCuda->store);

		cudaMalloc(&decListMap, decCount * sizeof ull);
		cudaMalloc(&fracListMap, decCount * sizeof fraction_list);
		cudaMalloc(&fracListCuda, fracCount * sizeof pk_list);

		int i = 0;
		for (std::list<DecimalRecord>::iterator it = dimension->begin(); it != dimension->end(); ++it)
		{
			decListMap[i] = it->decimal;
			cudaMemcpy(fracListMap[i], it->fractions, fracCount * sizeof pk_list, cudaMemcpyHostToDevice);
			i++;
		}

		cudaMemcpy(decListMap, decList, decCount * sizeof ull, cudaMemcpyHostToDevice);
		cudaMemcpy(fracListMap, fracList, decCount * sizeof fraction_list, cudaMemcpyHostToDevice);
		for (i--; i >= 0; i--)
		{
			cudaMemcpy(fracListMap[i], dimension->fra , fracCount * sizeof fraction_list, cudaMemcpyHostToDevice);
		}
	}

	void decimate(StoreCuda * storeCuda)
	{

	}

}