#pragma once
#include <list>
#include <math.h>
#include <algorithm>
#include <string>
#include <regex>

typedef unsigned long long ull;

typedef unsigned long int primary_key;
typedef primary_key* pk_list;
typedef pk_list* fraction_list;

struct DecimalRecord
{
	ull decimal;
	fraction_list fractions;
};

typedef std::list<DecimalRecord> decimal_record_list;


namespace Store 
{
	struct Store
	{
		int fragmentationLevel;
		decimal_record_list x;
		decimal_record_list y;
		decimal_record_list z;
		primary_key primaryKey;
	};

	Store * init();

	void setFragmentationLevel(Store * store, int fragmentationLevel);

	void addPointXYZ(Store * store, double x, double y, double z);

	void addDimension(Store * store, std::list<DecimalRecord> * dimension, double value, primary_key primaryKey);

	std::string getFractionString(Store * store, double number);
	std::string getFractionString(Store * store, const std::string number);

	int getNthFraction(Store * store, double number, int position);
	int getNthFraction(Store * store, std::string number, int position);

	DecimalRecord * getDecimalRecordFor(Store * store, decimal_record_list * dimension, ull decimalValue);

	ull calcFracListLength(Store * store);

}
