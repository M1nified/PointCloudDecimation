#pragma once

#include "Types.h"
#include <algorithm>

meter getMeter(dimension dim);
rest getRest(dimension dim);

namespace Storage
{
	bool init(Store * store);
	bool addPoint(Store * store, Point * point);
	bool assignPointToMeter(Store * store, Point * point, char dimCode);
	Meter * getMeterStruct(Store * store, meter_list * ml, meter m);
	dimension getSegmentIndex(Store * store, dimension dim);
	bool addPointToMeter(Store * store, Meter * meter, dimension dim, Point * point);
	bool setD(Store * store, dimension d);
}
