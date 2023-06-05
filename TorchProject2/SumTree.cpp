#include "SumTree.h"
#include <iostream>

SumTree::SumTree(int64_t size) :
	Tree(size)
{}

void SumTree::update_(int64_t index, float pe) {
	float diff = pe - nodes[index];
	nodes[index] = pe;

	while (index != 0) {
		index = (index - 1) / 2;
		nodes[index] += diff;
	}
}

int64_t SumTree::sample(float pe) {
	int64_t node(0), child1, child2;
	float res;
	while (true) {
		child1 = 2 * node + 1;
		child2 = child1 + 1;

		if (child1 >= nodes.size()) {
			break;
		}

		if (pe <= nodes[child1]) {
			node = child1;
		}
		else {
			pe -= nodes[child1];
			node = child2;
		}
	}
	return abs2rel(node);
}

std::vector<int64_t> SumTree::sample_batch(int64_t batch_size) {
	std::vector<int64_t> indexes(batch_size);

	float segment = nodes[0] / batch_size;
	float rnd;
	for (size_t i = 0; i < batch_size; i++) {
		rnd = (float)(std::rand()) / (float)(RAND_MAX);
		rnd = segment * (rnd + (float)i);
		indexes[i] = sample(rnd);
	}
	return indexes;
}