// DecimateCPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <list>

#include "Types.h"
#include "Storage.h"
#include "Filter.h"

int main()
{
	
	dimension d = 30;

	//Point ** points = NULL;

	Store store;
	Storage::init(&store);
	Storage::setD(&store, d);
	Storage::setInputPtsFile(&store, "R:\SONGA_BREEZE_L4.pts");

	if (!Storage::loadFromPtsFile(&store))
	{
		return 1;
	}

	SFilter filter;
	Filter::init(&filter, &store);
	Filter::setOutputFilename(&filter, "D:\filteredPoints.pts");
	Filter::filter(&filter);
	Filter::exportToPtsFile(&filter);
	
	return 0;
}