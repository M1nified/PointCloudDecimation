#pragma once

#include "Point.h"

#include <string>
#include <fstream>
#include <iostream>

class TxtToBinConverter
{
private:
	std::string textFileName;
	std::string binFileName;

	int pointBufferSize = 1000000;
	int maxDimensionValueInMm = 1000000;

public:
	TxtToBinConverter();
	~TxtToBinConverter();

	void SetTextFileName(std::string fileName);
	void SetBinFileName(std::string fileName);

	int Convert();
};

