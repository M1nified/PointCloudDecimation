#include "stdafx.h"
#include "PointCloudDecimator.h"
#include <fstream>
#include <string>


PointCloudDecimator::PointCloudDecimator()
{
}


PointCloudDecimator::~PointCloudDecimator()
{
}

bool PointCloudDecimator::AddPoint(Point point)
{
	return false;
}

bool PointCloudDecimator::LoadFromPTSFile(std::string filename)
{
	std::ifstream ifs(filename);
	std::string size = "";
	std::string line = "";
	Point * point;
	if (ifs.is_open())
	{
		if (getline(ifs, size))
		{
			this->initialSize = std::stoi(size);
		}
		for (; getline(ifs, line);)
		{
			point = new Point();
			point->Parse(line);
			this->AddPoint(*point);
		}
		ifs.close();
	}
	return false;
}
