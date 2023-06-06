#pragma once

#include <torch/torch.h>


class LinearLRS : public torch::optim::LRScheduler {
public:
    LinearLRS(
        torch::optim::Optimizer& optimizer,
        const unsigned step_size,
        const double start_factor,
        const double end_factor,
        const unsigned steps);

private:
    std::vector<double> get_lrs() override;

    const unsigned step_size;
    const double start_factor, end_factor;
    const double diff_;
    std::vector<double> lrs_base_;
    double factor_;
};
