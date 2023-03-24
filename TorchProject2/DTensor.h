#pragma once

#include <ATen/ATen.h>
#include <stdint.h>
#include <c10/core/DefaultDtype.h>

class DTensor {
public:
	DTensor();
	DTensor(at::IntArrayRef size, at::TensorOptions& options);

	at::Tensor get(int64_t index);

	void push(at::Tensor value);

	at::Tensor DTensor::index(at::Tensor indices);

private:
	int64_t start = 0;
	at::Tensor tensor;
};