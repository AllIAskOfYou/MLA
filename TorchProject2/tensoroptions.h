#pragma once

#include <ATen/ATen.h>
#include <c10/core/DefaultDtype.h>

namespace opt {
	auto Float32 = at::TensorOptions().dtype(c10::ScalarType::Float);
	auto Int64 = at::TensorOptions().dtype(c10::ScalarType::Long);
}
