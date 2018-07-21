#pragma once

#include "Types.h"
#include "Storage.h"

#include <string>
#include <fstream>
#include <stdio.h>

#define EOL_STRING "\r\n"

namespace Filter
{
	bool init(SFilter * filter, Store * store);
	void filterDeleteCollisionBeforeGroup(SFilter * filter, meter_list * metList, Meter * meter, Point * point);
	void filterDeleteCollisionsAfterGroup(SFilter * filter, meter_list * metList, Meter * meter, Point * point);
	void filterDeleteCollisionsAfterSegment(SFilter * filter, PointArr * theAfterSegment, Point * point);
	bool filterBySingleDim(SFilter * filter, meter_list * singleDimList);
	bool filter(SFilter * filter);
	bool setOutputFilename(SFilter * filter, std::string filename);
	bool areColliding(SFilter * filter, Point * pointA, Point * pointB);
	bool exportToPtsFile(SFilter * filter);
}