//
// Created by escha on 16.07.25.
//

#include <stddef.h>

#include "gradients.h"

Gradient mse_gradient(const Data *data, const Model *model)
{
    Gradient grad = {0};
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        const double pred = sample->features[0] * model->weights[1] + model->weights[0];
        const double error = pred - sample->label;
        grad.values[0] += error;
        grad.values[1] += error * sample->features[0];
    }
    grad.values[0] *= 2.0 / data->num_samples;
    grad.values[1] *= 2.0 / data->num_samples;

    return grad;
}
