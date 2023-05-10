#pragma once

#include <ATen/ATen.h>

typedef struct State {
	at::Tensor eStates;		// env state
	at::Tensor aStates;		// agent state
	at::Tensor oStates;		// oponent state
	at::Tensor aActions;	// agent action
	at::Tensor oActions;	// oponent action
} State;