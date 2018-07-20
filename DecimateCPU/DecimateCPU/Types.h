#pragma once
#include <list>

typedef unsigned long long ull;

typedef int dimension;

typedef short int meter;
typedef unsigned short int rest;

struct Point
{
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
	meter_list xDim, yDim, zDim;
};