#pragma once

namespace znn {
struct BatchNormOptions {
    double eps;
    double momentum;
    bool training;
};
}
