// DecimateCPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <list>

#include "Types.h"
#include "Storage.h"

bool parseLineToPoint(Point * point, char * input);
bool loadFromPTSFile(Store * store, std::string filename, ull * initialSize, Point *** cloudArr);


int main()
{
	
	dimension d = 30;

	ull initialSize;
	Point ** points = NULL;

	Store store;
	Storage::setD(&store, d);

	if (!loadFromPTSFile(&store, "R:\SONGA_BREEZE_L4.pts", &initialSize, &points))
	{
		return 1;
	}
	return 0;
}

dimension meterToDimension(double meter)
{
	dimension dim = (dimension)floor(meter * 10000);
	return dim;
}

bool parseLineToPoint(Point * p, char * line)
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

bool loadFromPTSFile(Store * store, std::string filename, ull * initialSize, Point *** cloudArrr)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	double x, y, z;

	Point ** cloudArr;

	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			*initialSize = std::stoull(size);
			cloudArr = (Point **)malloc(*initialSize * sizeof (Point*));
			for (ull i = 0; getline(ifs, line) && i<10000; i++)
			{
				const char * l = line.c_str();
				cloudArr[i] = new Point();
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
