#pragma once
#include <list>
#include <vector>

typedef unsigned long long ull;

typedef int dimension;

typedef short int meter;
typedef unsigned short int rest;

struct Point
{
	bool exists;
	dimension x, y, z;
};

typedef std::vector<Point*> segments;

struct Meter
{
	meter meter;
	segments * segments;
};

typedef std::list<Meter *> meter_list;

struct Store
{
	dimension d;
	ull segmentsCount;
	meter_list xDim, yDim, zDim;
	ull initialSize;
	Point ** points;
	std::string inputFilename;
};

struct SFilter
{
	Store * store;
	std::string outputFilename;
	ull deletedCount;
	ull storedCount;
};