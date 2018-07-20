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
		for (int i = 0; i < segCount; i++)
		{
			bool oneLeftAlive = false;
			for (int j = 0; j < (*it)->segments[i].count; j++)
			{
				if (oneLeftAlive)
				{
					(*it)->segments[i].points[j].exists = false;
					filter->deletedCount++;
				}
				else if ((*it)->segments[i].points[j].exists)
				{
					oneLeftAlive = true;
					Point * point = &(*it)->segments[i].points[j];
					if (i == 0)
					{
						filterDeleteCollisionBeforeGroup(filter, xMetList, *it, point);
					}
					else if (i == filter->store->segmentsCount - 1)
					{
						filterDeleteCollisionsAfterGroup(filter, xMetList, *it, point);
					}
					else
					{
						filterDeleteCollisionsAfterSegment(filter, &(*it)->segments[i+1], point);
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

	std::ifstream ifs (store->inputFilename);
	std::ofstream ofs (filter->outputFilename);


	/*try
	{
		ifs.open(store->inputFilename, std::ifstream::in);
		ofs.open(filter->outputFilename, std::ofstream::out | std::ofstream::trunc);
	}
	catch (std::ios_base::failure &e)
	{
		std::cerr << e.what() << EOL_STRING;
		return false;
	}
	catch (...)
	{
		return false;
	}*/
	auto finalPointCount = store->initialSize - filter->deletedCount;

	if (ifs.good() && ofs.good() && ifs.is_open() && ofs.is_open())
	{
		ofs << finalPointCount << EOL_STRING;
		getline(ifs, inLine);
		for (int i = 0; i < store->initialSize && getline(ifs, inLine); i++)
		{
			if (points[i]->exists)
			{
				ofs << inLine;
			}
			delete points[i];
		}
		ifs.close();
		ofs.close();
	}
	return true;
}
