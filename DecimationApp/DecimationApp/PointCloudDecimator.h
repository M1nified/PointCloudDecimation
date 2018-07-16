#pragma once
class PointCloudDecimator
{
public:
	PointCloudDecimator();
	~PointCloudDecimator();

	void Decimate();
	void SetDecimationThreshold(float threshold);

	bool AddPoint(Point point);
	bool LoadFromPTSFile(std::string filename);

private:
	std::list<Point> cloud;
	float threshold;
};

