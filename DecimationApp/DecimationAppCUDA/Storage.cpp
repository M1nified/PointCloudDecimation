#include "Storage.h"
namespace Store {
	Store * init()
	{
		Store * store = new Store();
		return store;
	}
	void setFragmentationLevel(Store * store, int fragmentationLevel)
	{
		store->fragmentationLevel = fragmentationLevel;
	}
	void addPointXYZ(Store * store, double x, double y, double z)
	{
		addDimension(&(store->x), x);
		addDimension(&(store->y), y);
		addDimension(&(store->z), z);
	}
	void addDimension(std::list<DecimalRecord> * dimension, double value)
	{
		double fract, dec;
		DecimalRecord * decimalRecord;
		fract = modf(value, &dec);
		decimalRecord = getDecimalRecordFor(dimension, (unsigned long long)(dec + 0.5));
	}
	DecimalRecord * getDecimalRecordFor(std::list<DecimalRecord>* dimension, unsigned long long decimalValue)
	{
		DecimalRecord * result = nullptr;
		for (auto it = dimension->begin(); it != dimension->end(); ++it)
		{
			if ((*it).decimal == decimalValue) 
			{
				result = &(*it);
				break;
			}
		}
		if (result == nullptr)
		{
			DecimalRecord * dr = new DecimalRecord();
			dr->decimal = decimalValue;
			dimension->push_back(*dr);
			result = dr;
		}
		return result;
	}
}