// DecimationApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PointCloudDecimator.h"
#include <fstream>
#include "Point.h"

bool loadFromPTSFile(std::string filename, int * initialSize);

int main()
{
	//PointCloudDecimator * decimator = new PointCloudDecimator();
	//decimator->LoadFromPTSFile("SONGA_BREEZE_L4.pts");
	int initialSize;
	Point * cloud;
	loadFromPTSFile("SONGA_BREEZE_L4.pts", &initialSize);
    return 0;
}

bool loadFromPTSFile(std::string filename, int * initialSize)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	Point * point;
	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			* initialSize = std::stoi(size);
		}
		for (; getline(ifs, line);)
		{
			point = new Point();
			point->Parse(line);
			//this->AddPoint(*point);
			printf_s("%s\n", line.c_str());
		}
		ifs.close();
	}
	return false;
}
