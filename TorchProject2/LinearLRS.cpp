#include "LinearLRS.h"

LinearLRS::LinearLRS(
    torch::optim::Optimizer& optimizer,
    const unsigned step_size,
    const double start_factor,
    const double end_factor,
    const unsigned steps
) :
    LRScheduler(optimizer),
    step_size(step_size),
    start_factor(start_factor),
    end_factor(end_factor),
    diff_((start_factor - end_factor) / (float)(steps / step_size)),
    factor_(start_factor),
    lrs_base_(get_current_lrs())
{}

std::vector<double> LinearLRS::get_lrs() {
    std::vector<double> lrs(lrs_base_.size());
    if (step_count_ % step_size == 0) {
        factor_ -= diff_;
        factor_ = factor_ < end_factor ? end_factor : factor_;
        //std::cout << "Factor: " << factor_ << "diff: " << diff_ << std::endl;
    }
    for (int i = 0; i < lrs.size(); i++) {
        lrs[i] = factor_ * lrs_base_[i];
    }
    return lrs;
}