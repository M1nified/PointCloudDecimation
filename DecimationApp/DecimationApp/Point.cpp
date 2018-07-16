#include "stdafx.h"
#include "Point.h"
#include "algorithm"

Point::Point()
{
}


Point::~Point()
{
}

double Point::GetCubicDistanceTo(Point point)
{
	double diff = 0.0f;
	diff = std::max(this->x - point.x, this->y - point.y);
	diff = std::max(diff, this->z - point.z);
	return diff;
}

double Point::GetDistanceTo(Point point)
{
	double diff = this->GetCubicDistanceTo(point);
	return diff;
}

void Point::SetXYZ(double x, double y, double z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

bool Point::Parse(std::string input)
{
	const char * in = input.c_str();
	return this->Parse((char *)in);
}

bool Point::Parse(char * input)
{
	double x, y, z;
	if (3 == sscanf_s(input, "%lf %lf %lf%*s", &x, &y, &z))
	{
		this->SetXYZ(x, y, z);
		return true;
	}
	return false;
}
