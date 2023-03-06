#include "DTensor.h"

using namespace at::indexing;

DTensor::DTensor() {}

/*
DTensor::DTensor(int64_t maxSize, int64_t dim, int64_t lastN, at::TensorOptions& options) :
	maxSize(maxSize)
{
	tensor = at::zeros({ maxSize, dim, lastN }, options);
	a = tensor;
}
*/

DTensor::DTensor(at::IntArrayRef size, at::TensorOptions& options) :
	size(size)
{
	tensor = at::zeros(size, options);
	a = tensor;
}

at::Tensor DTensor::get(int64_t index) {
	return tensor.index({ (start + index) % size[0] });
}

void DTensor::push(at::Tensor value) {
	tensor.index({ start, Slice(), Slice(0, -1) }) = tensor.index({ start - 1, Slice(), Slice(1) });
	tensor.index({ start, Slice(), -1 }) = value;
	start = (start + 1) % size[0];
	a = tensor.roll(-start, { 0 }).squeeze(1);
}

at::Tensor DTensor::toTensor() {
	return tensor.roll(-start, { 0 });
}
