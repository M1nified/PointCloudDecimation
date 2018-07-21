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
	store->inputFilename = "";
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

Meter * Storage::getMeterStruct(Store * store, meter_list * ml, meter m, Meter * meter)
{
	if (meter == nullptr)
	{
		meter = findMeterStruct(store, ml, m);
	}
	if (meter == nullptr)
	{
		meter = new Meter();
		meter->meter = m;
		auto count = store->segmentsCount;
		meter->segments = (PointArr *)malloc(count * sizeof PointArr);
		for (int i = 0; i < count; i++)
		{
			meter->segments[i].count = (ull)0;
		}
		ml->push_front(meter);
	}
	return meter;
}

Meter * Storage::findMeterStruct(Store * store, meter_list * ml, meter m)
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
	auto devider = 10 * DEVIDER;
	auto count = devider / store->d;
	store->segmentsCount = count;
	return true;
}

bool Storage::setInputPtsFile(Store * store, std::string filename)
{
	store->inputFilename = filename;
	return true;
}

bool Storage::parseLineToPoint(Point * p, char * line)
{
	double x, y, z;
	if (3 == sscanf_s(line, "%lf %lf %lf%*s", &x, &y, &z))
	{
		p->x = meterToDimension(x);
		p->y = meterToDimension(y);
		p->z = meterToDimension(z);
		return true;
	}
	return false;
}

dimension Storage::meterToDimension(double meter)
{
	dimension dim = (dimension)floor(meter * 10000);
	return dim;
}

bool Storage::loadFromPtsFile(Store * store)
{
	static std::ifstream ifs(store->inputFilename);
	std::string size = "";
	std::string line = "";
	double x, y, z;

	Point ** cloudArr;

	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			store->initialSize = std::stoull(size);
			//store->initialSize = 1000;
			cloudArr = (Point **)malloc(store->initialSize * sizeof(Point*));
			store->points = cloudArr;
			for (ull i = 0; getline(ifs, line) && i < store->initialSize; i++)
			{
				const char * l = line.c_str();
				cloudArr[i] = new Point();
				cloudArr[i]->exists = true;
				if (parseLineToPoint(cloudArr[i], (char *)l))
				{
					Storage::addPoint(store, cloudArr[i]);
					//cloudArr[i] = &p;
				}
			}
			ifs.close();
			return true;
		}
		ifs.close();
	}
	return false;
}
