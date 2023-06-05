#pragma once

#include <vector>
#include <stdint.h>
#include "ATen/ATen.h"

class Tree {
public:
	Tree(int64_t size);

	void update(int64_t index, float pe);

	void push(float pe);

	void update_batch(
		std::vector<int64_t> indexes,
		std::vector<float> pes
	);
	void update_batch(at::Tensor indexes, at::Tensor pes);

	std::vector<float> get(std::vector<int64_t> indexes);
	at::Tensor get(at::Tensor indexes);

	float get_value();

	int64_t rel2abs(int64_t index);
	int64_t abs2rel(int64_t index);

protected:
	virtual void update_(int64_t index, float pe) = 0;

protected:
	int64_t size;
	std::vector<float> nodes;
	int64_t start = 0;
};
