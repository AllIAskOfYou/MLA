#include "Tree.h"

Tree::Tree(int64_t size) :
	size(size)
{
	nodes = std::vector<float>(2 * size - 1, 0);
}

void Tree::push(float pe) {
	update(0, pe);

	start = start + 1;
	if (start == size) {
		start = 0;
	}
}

void Tree::update(int64_t index, float pe) {
	index = rel2abs(index);
	update_(index, pe);
}

void Tree::update_batch(
	std::vector<int64_t> indexes,
	std::vector<float> pes
)
{
	for (size_t i = 0; i < indexes.size(); i++) {
		update(indexes[i], pes[i]);
	}
}

std::vector<float> Tree::get(std::vector<int64_t> indexes) {
	std::vector<float> pes(indexes.size());
	for (size_t i = 0; i < indexes.size(); i++) {
		pes[i] = nodes[rel2abs(indexes[i])];
	}
	return pes;
}

float Tree::get_value() {
	return nodes[0];
}

int64_t Tree::abs2rel(int64_t index) {
	float res;
	res = (index - (size - 1) - start) % size;
	res = res < 0 ? res + size : res;
	return res;
}

int64_t Tree::rel2abs(int64_t index) {
	float res;
	res = (index + start) % size;
	res = res < 0 ? res + size : res;
	res += size - 1;
	return res;
}