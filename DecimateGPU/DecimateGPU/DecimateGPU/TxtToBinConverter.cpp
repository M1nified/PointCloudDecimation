#include "TxtToBinConverter.h"



TxtToBinConverter::TxtToBinConverter()
{
}


TxtToBinConverter::~TxtToBinConverter()
{
}

void TxtToBinConverter::SetTextFileName(std::string fileName)
{
	this->textFileName = fileName;
}

void TxtToBinConverter::SetBinFileName(std::string fileName)
{
	this->binFileName = fileName;
}

int TxtToBinConverter::Convert()
{
	std::ifstream ifs;
	ifs.open(this->textFileName, std::ios::binary);
	FILE * output = fopen(this->binFileName.c_str(), "wb");

	if (!ifs.is_open())
	{
		fclose(output);
		return 1;
	}
	if (output == NULL)
	{
		ifs.close();
		return 2;
	}

	std::string sizeStr;
	if (getline(ifs, sizeStr))
	{
		float x, y, z;
		auto len = this->pointBufferSize;
		Point * p = (Point *)malloc(len * sizeof(struct Point));
		int i = 0;
		while (ifs >> x >> y >> z >> p[i].prop1 >> p[i].prop2 >> p[i].prop3 >> p[i].prop4)
		{
			if (x > this->maxDimensionValueInMm || y > this->maxDimensionValueInMm || z > this->maxDimensionValueInMm)
			{
				continue;
			}
			p[i].x = x * 1000;
			p[i].y = y * 1000;
			p[i].z = z * 1000;

			if (i == len - 1)
			{
				std::cout << i << std::endl;
				fwrite(p, sizeof(struct Point), len, output);
				i = 0;
			}
			else
			{
				i++;
			}
		}
		if (i < len - 1)
		{
			fwrite(&p, sizeof(struct Point), i + 1, output);
		}
	}
	fclose(output);
	ifs.close();

	return 0;
}
