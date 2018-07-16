#include "stdafx.h"
#include "Point.h"
#include "algorithm"
#include <string>

Point::Point()
{
}


Point::~Point()
{
}

float Point::GetCubicDistanceTo(Point point)
{
	float diff = 0.0f;
	diff = std::max(this->x - point.x, this->y - point.y);
	diff = std::max(diff, this->z - point.z);
	return diff;
}
