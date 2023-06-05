#pragma once

#include "Tree.h"

class MaxTree : public Tree {
public:
	MaxTree(int64_t size);

	void update_(int64_t index, float pe);
};