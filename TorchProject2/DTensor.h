#pragma once

#include <ATen/ATen.h>
#include <stdint.h>

class DTensor {
public:
	DTensor();
	DTensor(int64_t maxSize, int64_t dim, int64_t lastN, at::TensorOptions& options);

	at::Tensor get(int64_t index);

	void push(at::Tensor value);

	at::Tensor toTensor();

private:
	int64_t start = 0;
	int64_t maxSize = 0;
	at::Tensor tensor;
public:
	at::Tensor a;
};