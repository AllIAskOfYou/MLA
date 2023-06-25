#pragma once

#include "ATen/ATen.h"

struct ICMReturn {
	at::Tensor aPred;
	at::Tensor sNext;
	at::Tensor sNextPred;
};