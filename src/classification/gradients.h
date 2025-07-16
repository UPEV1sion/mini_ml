//
// Created by escha on 16.07.25.
//

#pragma once

#include "ml/ml.h"

Gradient log_loss_gradient(const Data *data, const Model *model);
double zero_one_loss(const Data *data, const Model *model);