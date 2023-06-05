#pragma once

#include <vector>
#include <stdint.h>

class Tree {
public:
	Tree(int64_t size);

	void update(int64_t index, float pe);
	virtual void update_(int64_t index, float pe) = 0;

	void push(float pe);
	void update_batch(
		std::vector<int64_t> indexes,
		std::vector<float> pes
	);
	std::vector<float> get(std::vector<int64_t> indexes);
	float get_value();

	int64_t rel2abs(int64_t index);
	int64_t abs2rel(int64_t index);

protected:
	int64_t size;
	std::vector<float> nodes;
	int64_t start = 0;
};
