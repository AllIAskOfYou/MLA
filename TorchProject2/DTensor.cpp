#include "DTensor.h"

using namespace at::indexing;

DTensor::DTensor() {}

DTensor::DTensor(at::IntArrayRef size, at::TensorOptions& options)
{
	tensor = at::zeros(size, options);
}
at::Tensor DTensor::get(int64_t index) {
	return tensor.index({ (start + index) % tensor.size(0) });
}

void DTensor::push(at::Tensor value) {
	tensor.index({ start, Slice(0, -1) }) = tensor.index({ start - 1, Slice(1) });
	tensor.index({ start, Slice(-1) }) = value;
	start = (start + 1) % tensor.size(0);
}

at::Tensor DTensor::index(at::Tensor indices) {
	return tensor.index({ (indices + start) % tensor.size(0) });
}