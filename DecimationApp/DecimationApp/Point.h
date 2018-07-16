#pragma once
class Point
{
public:
	float x;
	float y;
	float z;

	Point();
	~Point();

	float GetCubicDistanceTo(Point point);
	float GetDistanceTo(Point point);
};

