#include "MaxTree.h"
#include <iostream>

MaxTree::MaxTree(int64_t size) :
	Tree(size)
{}


void MaxTree::update_(int64_t index, float pe) {
	int64_t child1, child2;
	float max;
	index = ((index + start) % size) + size - 1;
	nodes[index] = pe;

	while (index != 0) {
		index = (index - 1) / 2;
		child1 = 2 * index + 1;
		child2 = child1 + 1;
		max = std::max<float>(nodes[child1], nodes[child2]);
		if (nodes[index] == max) {
			break;
		}
		nodes[index] = max;
	}
}