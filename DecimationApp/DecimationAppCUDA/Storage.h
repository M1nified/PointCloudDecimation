#pragma once
#include <list>
#include <math.h>
#include <algorithm>

struct DecimalRecord
{
	unsigned long long decimal;
	void * records [10];
};


namespace Store 
{
	struct Store
	{
		int fragmentationLevel;
		std::list<DecimalRecord> x;
		std::list<DecimalRecord> y;
		std::list<DecimalRecord> z;
	};

	Store * init();
	void setFragmentationLevel(Store * store, int fragmentationLevel);
	void addPointXYZ(Store * store, double x, double y, double z);
	void addDimension(std::list<DecimalRecord> * dimension, double value);
	DecimalRecord * getDecimalRecordFor(std::list<DecimalRecord> * dimension, unsigned long long decimalValue);
}
