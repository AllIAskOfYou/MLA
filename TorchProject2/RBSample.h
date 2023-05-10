#pragma once

#include <ATen/ATen.h>
#include "State.h"

typedef struct RBSample {
	State states;			// states				(t)
	at::Tensor aActions;	// agent actions		(t)
	at::Tensor rewards;		// rewards				(t)
	State nStates;			// next states			(t+1)
} RBSample;