#pragma once
class PointCloudDecimator
{
public:
	PointCloudDecimator();
	~PointCloudDecimator();

	void Decimate();
	void SetDecimationThreshold(float threshold);

private:
	std::list<Point> cloud;
	float threshold;
};

