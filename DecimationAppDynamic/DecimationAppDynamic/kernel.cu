
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <list>

typedef unsigned long long ull;

typedef short int dim_dec;
typedef unsigned int dim_frac;

struct Dimension
{
	dim_dec dec;
	dim_frac frac;
};

struct Point
{
	Dimension x, y, z;
};

typedef std::list<Point> point_list;

bool parseLineToPoint(Point * point, char * input);
bool loadFromPTSFile(std::string filename, ull * initialSize, Point ** cloudArr);


int main()
{
	double d = 0.00005;
	ull initialSize;
	Point * points;
	if (!loadFromPTSFile("R:\SONGA_BREEZE_L4.pts", &initialSize, &points))
	{
		return 1;
	}
	return 0;
}

bool parseLineToPoint(Point * p, char * line)
{
	dim_dec xd, yd, zd;
	dim_frac xf, yf, zf;
	if (6 == sscanf_s(line, "%hu.%u %hu.%u %hu.%u%*s", &xd, &xf, &yd, &yf, &zd, &zf))
	{
		p->x.dec = xd;
		p->x.frac = xf;
		p->y.dec = yd;
		p->y.frac = yf;
		p->z.dec = zd;
		p->z.frac = zf;
		return true;
	}
	return false;
}

bool loadFromPTSFile(std::string filename, ull * initialSize, Point ** clourArr)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	double x, y, z;

	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			*initialSize = std::stoull(size);
			*clourArr = (Point *)malloc(*initialSize * sizeof Point);
			for (ull i = 0; getline(ifs, line); i++)
			{
				const char * l = line.c_str();
				Point p;
				if (parseLineToPoint(&p, (char *)l))
				{
					(*clourArr)[i] = p;
				}
			}
			ifs.close();
			return true;
		}
		ifs.close();
	}
	return false;
}
