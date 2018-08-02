#pragma once

#include "Types.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <stdlib.h>

meter getMeter(dimension dim);
rest getRest(dimension dim);

namespace Storage
{
	bool init(Store * store);
	bool addPoint(Store * store, Point * point);
	bool assignPointToMeter(Store * store, Point * point, char dimCode);
	Meter * getMeterStruct(Store * store, meter_list * ml, meter m, Meter * meter = nullptr);
	Meter * findMeterStruct(Store * store, meter_list * ml, meter m);
	dimension getSegmentIndex(Store * store, dimension dim);
	bool addPointToMeter(Store * store, Meter * meter, dimension dim, Point * point);
	bool setD(Store * store, dimension d);
	bool setInputPtsFile(Store * store, std::string filename);
	bool parseLineToPoint(Point * p, char * line);
	dimension meterToDimension(double meter);
	bool loadFromPtsFile(Store * store);
}
