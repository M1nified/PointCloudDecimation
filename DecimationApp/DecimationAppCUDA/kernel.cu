#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <list>

#include "Point.h"
#include "Storage.h"




bool loadFromPTSFile(std::string filename, unsigned long long * initialSize, Store::Store ** store);

int main()
{
	unsigned long long initialSize;

	Store::Store * store = Store::init();
	Store::setFragmentationLevel(store, 5);

	loadFromPTSFile("SONGA_BREEZE_L4.pts", &initialSize, &store);
	return 0;
}

bool loadFromPTSFile(std::string filename, unsigned long long * initialSize, Store::Store ** store)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	Point * point;
	point = new Point();
	double x, y, z;

	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			*initialSize = std::stoull(size);
			for (unsigned long long i = 0; getline(ifs, line) && i < 100000; i++)
			{
				point->Parse(line);
				x = point->x;
				y = point->y;
				z = point->z;
				Store::addPointXYZ(*store, x, y, z);
			}
			ifs.close();
			return true;
		}
		ifs.close();
	}
	return false;
}
