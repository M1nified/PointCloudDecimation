#include "stdafx.h"
#include "Filter.h"

bool Filter::init(SFilter * filter, Store * store)
{
	filter->outputFilename = "";
	filter->deletedCount = 0;
	filter->store = store;
	return true;
}

void Filter::filterDeleteCollisionBeforeGroup(SFilter * filter, meter_list * metList, Meter * meter, Point * point)
{
	Meter * previousMeter = Storage::findMeterStruct(filter->store, metList, meter->meter - 1);
	if (previousMeter != nullptr)
	{
		previousMeter = Storage::getMeterStruct(filter->store, metList, previousMeter->meter - 1, previousMeter);
		PointArr * last = &previousMeter->segments[filter->store->segmentsCount - 1];
		if (last->count > 0)
		{
			for (int j = 0; j < last->count; j++)
			{
				if (last->points[j].exists && areColliding(filter, point, &last->points[j]))
				{
					last->points[j].exists = false;
					filter->deletedCount++;
				}
			}

		}
	}
}

void Filter::filterDeleteCollisionsAfterGroup(SFilter * filter, meter_list * metList, Meter * meter, Point * point)
{
	Meter * nextMeter = Storage::findMeterStruct(filter->store, metList, meter->meter + 1);
	if (nextMeter != nullptr)
	{
		nextMeter = Storage::getMeterStruct(filter->store, metList, nextMeter->meter + 1, nextMeter);
		PointArr * first = &nextMeter->segments[0];
		if (first->count > 0)
		{
			for (int j = 0; j < first->count; j++)
			{
				if (first->points[j].exists && areColliding(filter, point, &first->points[j]))
				{
					first->points[j].exists = false;
					filter->deletedCount++;
				}
			}

		}
	}
}

void Filter::filterDeleteCollisionsAfterSegment(SFilter * filter, PointArr * theAfterSegment, Point * point)
{
	if (theAfterSegment->count > 0)
	{
		for (int j = 0; j < theAfterSegment->count; j++)
		{
			if (theAfterSegment->points[j].exists && areColliding(filter, point, &theAfterSegment->points[j]))
			{
				theAfterSegment->points[j].exists = false;
				filter->deletedCount++;
			}
		}
	}
}

bool Filter::filter(SFilter * filter)
{
	meter_list * xMetList = &filter->store->xDim;
	for (auto it = xMetList->begin(); it != xMetList->end(); ++it)
	{
		auto segCount = filter->store->segmentsCount;
		for (int segmentIndex = 0; segmentIndex < segCount; segmentIndex++)
		{
			bool oneLeftAlive = false;
			for (int j = 0; j < (*it)->segments[segmentIndex].count; j++)
			{
				if (oneLeftAlive)
				{
					(*it)->segments[segmentIndex].points[j].exists = false;
					filter->deletedCount++;
				}
				else if ((*it)->segments[segmentIndex].points[j].exists)
				{
					oneLeftAlive = true;
					Point * point = &(*it)->segments[segmentIndex].points[j];
					if (segmentIndex == 0)
					{
						filterDeleteCollisionBeforeGroup(filter, xMetList, *it, point);
					}
					else if (segmentIndex == filter->store->segmentsCount - 1)
					{
						filterDeleteCollisionsAfterGroup(filter, xMetList, *it, point);
					}
					else
					{
						filterDeleteCollisionsAfterSegment(filter, &(*it)->segments[segmentIndex+1], point);
					}
				}
			}

		}
	}
	return true;
}

bool Filter::setOutputFilename(SFilter * filter, std::string filename)
{
	filter->outputFilename = filename;
	return true;
}

bool Filter::areColliding(SFilter * filter, Point * pointA, Point * pointB)
{
	double distance = sqrt(
		pow((pointA->x - pointB->x), 2) +
		pow((pointA->y - pointB->y), 2) +
		pow((pointA->z - pointB->z), 2)
	);
	if (distance > filter->store->d)
	{
		return false;
	}
	return true;
}

bool Filter::exportToPtsFile(SFilter * filter)
{
	Store * store = filter->store;
	Point ** points = store->points;
	std::string inLine = "";

	if (filter->outputFilename.compare("") == 0 || store->inputFilename.compare("") == 0)
	{
		return false;
	}

	static std::fstream ifs(store->inputFilename, std::fstream::in);

	FILE * output;
	errno_t err;

	if ((err = fopen_s(&output, filter->outputFilename.c_str(), "w")) != 0)
	{
		return false;
	}

	auto finalPointCount = store->initialSize - filter->deletedCount;

	if (ifs.good() && ifs.is_open() && output != NULL)
	{
		fputs(finalPointCount + EOL_STRING, output);
		getline(ifs, inLine);
		for (int i = 0; i < store->initialSize && getline(ifs, inLine); i++)
		{
			if (points[i]->exists)
			{
				fputs(inLine.c_str(), output);
			}
			delete points[i];
		}
		ifs.close();
		fclose(output);
		return true;
	}
	return false;
}