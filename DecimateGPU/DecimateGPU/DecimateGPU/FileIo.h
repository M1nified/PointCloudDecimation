#pragma once

#include <fstream>

class FileIo
{
public:
	std::ifstream ifs;
	FileIo();
	~FileIo();
};

