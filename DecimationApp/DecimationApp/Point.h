#pragma once
#include <string>
class Point
{
public:
	double x;
	double y;
	double z;

	Point();
	~Point();

	double GetCubicDistanceTo(Point point);
	double GetDistanceTo(Point point);

	void SetXYZ(double y, double z, double x);

	bool Parse(std::string input);
	bool Parse(char * input);
};

