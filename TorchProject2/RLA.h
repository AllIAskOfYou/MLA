#pragma once

#include <ATen/ATen.h>

class RLA {
public:
	virtual void update() = 0;
	
	virtual at::Tensor nextAction() = 0;

private:

};