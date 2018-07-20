#include "stdafx.h"
#include "Storage.h"

#define DEVIDER 10000;

meter getMeter(dimension dim)
{
	meter m = dim / DEVIDER;
	return m;
}

rest getRest(dimension dim)
{
	rest r = abs(dim) % DEVIDER;
	return r;
}

bool Storage::init(Store * store)
{
	return true;
}

bool Storage::addPoint(Store * store, Point * point)
{
	assignPointToMeter(store, point, 'x');
	assignPointToMeter(store, point, 'y');
	assignPointToMeter(store, point, 'z');
	return true;
}

bool Storage::assignPointToMeter(Store * store, Point * point, char dimCode)
{
	meter_list * ml;
	dimension dim;
	meter m;
	Meter * meter;
	switch (dimCode)
	{
	case 'x':
		ml = &store->xDim;
		dim = point->x;
		break;
	case 'y':
		ml = &store->yDim;
		dim = point->y;
		break;
	default:
		ml = &store->zDim;
		dim = point->z;
		break;
	}

	m = getMeter(dim);
	meter = getMeterStruct(store, ml, m);
	addPointToMeter(store, meter, dim, point);
	return true;
}

Meter * Storage::getMeterStruct(Store * store, meter_list * ml, meter m)
{
	Meter * meter = nullptr;
	for (auto it = ml->begin(); it != ml->end(); ++it)
	{
		if ((*it)->meter == m)
		{
			meter = *it;
			break;
		}
	}
	if (meter == nullptr)
	{
		meter = new Meter();
		meter->meter = m;
		auto devider = 10 * DEVIDER;
		auto count = devider / store->d;
		meter->segments = (PointArr *)malloc(count * sizeof PointArr);
		for (int i = 0; i < count; i++)
		{
			meter->segments[i].count = (ull)0;
		}
		ml->push_front(meter);
	}
	return meter;
}

dimension Storage::getSegmentIndex(Store * store, dimension dim)
{
	dimension index = (getRest(dim) / store->d);
	return index;
}

bool Storage::addPointToMeter(Store * store, Meter * meter, dimension dim, Point * point)
{
	auto segmentIndex = getSegmentIndex(store, dim);
	PointArr * seg = &meter->segments[segmentIndex];
	if (seg->count == 0)
	{
		seg->points = (Point *)malloc(sizeof (Point*));
		seg->points[0] = *point;
	}
	else
	{
		Point * seg2 = (Point *)realloc(seg->points, seg->count * sizeof(Point *));
		seg->points = seg2;
	}
	seg->count++;
	return true;
}

bool Storage::setD(Store * store, dimension d)
{
	store->d = d;
	return true;
}
