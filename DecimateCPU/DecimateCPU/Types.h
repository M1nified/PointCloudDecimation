#pragma once
#include <list>

typedef unsigned long long ull;

typedef int dimension;

typedef short int meter;
typedef unsigned short int rest;

struct Point
{
	bool exists;
	dimension x, y, z;
};

struct PointArr
{
	ull count;
	Point * points;
};

struct Meter
{
	meter meter;
	PointArr * segments;
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