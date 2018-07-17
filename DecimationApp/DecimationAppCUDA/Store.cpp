#include "Store.h"
namespace Store {

	Store * init()
	{
		Store * store = new Store();
		store->primaryKey = 0;
		return store;
	}

	void setFragmentationLevel(Store * store, int fragmentationLevel)
	{
		store->fragmentationLevel = fragmentationLevel;
	}

	void addPointXYZ(Store * store, double x, double y, double z)
	{
		auto pk = store->primaryKey++;
		addDimension(store, &(store->x), x, pk);
		addDimension(store, &(store->y), y, pk);
		addDimension(store, &(store->z), z, pk);
	}

	void addDimension(Store * store, decimal_record_list * dimension, double value, primary_key primaryKey)
	{
		double fract, dec;
		DecimalRecord * decimalRecord;
		fract = modf(value, &dec);
		decimalRecord = getDecimalRecordFor(store, dimension, (ull)(dec + 0.5));
		int frac0 = getNthFraction(store, value, 0);
		auto fraction = getFractionString(store, value);
		unsigned long long int fractionIndex = std::stoull(fraction);
		auto pointsCountInLocation = decimalRecord->fractions[fractionIndex][0];
		decimalRecord->fractions[fractionIndex] = (pk_list)realloc(decimalRecord->fractions[fractionIndex], (pointsCountInLocation + 1) * sizeof primary_key);
		decimalRecord->fractions[fractionIndex][0] = pointsCountInLocation + 1;
		decimalRecord->fractions[fractionIndex][pointsCountInLocation] = primaryKey;
	}

	std::string getFractionString(Store * store, double number)
	{
		auto numberAsString = std::to_string(number);
		auto result = getFractionString(store, numberAsString);
		return result;
	}
	std::string getFractionString(Store * store, const std::string number)
	{
		std::regex rgx("[-\\+]?\\d+\\.(\\d*)");
		std::smatch m;
		if (std::regex_match(number.begin(), number.end(), m, rgx))
		{
			std::string frac = (std::string)m[1];
			frac.resize(store->fragmentationLevel, '0');
			return frac;
		}
		return std::string();
	}

	int getNthFraction(Store * store, double number, int position)
	{
		auto numberAsString = std::to_string(number);
		int result = getNthFraction(store, numberAsString, position);
		return result;
	}
	int getNthFraction(Store * store, const std::string number, int position)
	{
		auto frac = getFractionString(store, number);
		char elementAtPosition = frac[position];
		int element = elementAtPosition - '0';
		return element;
	}

	ull calcFracListLength(Store * store)
	{
		ull count = (ull)pow(10, store->fragmentationLevel);
		return count;
	}

	DecimalRecord * getDecimalRecordFor(Store * store, decimal_record_list* dimension, ull decimalValue)
	{
		DecimalRecord * result = nullptr;
		for (auto it = dimension->begin(); it != dimension->end(); ++it)
		{
			if ((*it).decimal == decimalValue)
			{
				result = &(*it);
				break;
			}
		}
		if (result == nullptr)
		{
			DecimalRecord * dr = new DecimalRecord();
			ull count = calcFracListLength(store);
			if (nullptr != (dr->fractions = (fraction_list)malloc(count * sizeof pk_list))) {

				for (int i = 0; i < count; i++)
				{
					if (nullptr != (dr->fractions[i] = (pk_list)calloc(5, sizeof primary_key)))
					{
						dr->fractions[i][0] = 1;
						auto x = dr->fractions[i][0];
					}
				}
				dr->decimal = decimalValue;
				dimension->push_back(*dr);
				result = dr;
			}
		}
		return result;
	}

}