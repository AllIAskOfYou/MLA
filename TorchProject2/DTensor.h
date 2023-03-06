#pragma once

#include <ATen/ATen.h>
#include <stdint.h>

class DTensor {
public:
	DTensor();
	DTensor(at::IntArrayRef size, at::TensorOptions& options);

	at::Tensor get(int64_t index);

	void push(at::Tensor value);

	at::Tensor toTensor();

private:
	int64_t start = 0;
	at::IntArrayRef size = 0;
	at::Tensor tensor;
public:
	at::Tensor a;
};