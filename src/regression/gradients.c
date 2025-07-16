//
// Created by escha on 16.07.25.
//

#include <stddef.h>

#include "gradients.h"

// 2/m sum_{i=1}^{m} x_i(<w, x_i> - y_i)
Gradient mse_gradient(const Data *data, const Model *model)
{
    Gradient grad = {0};
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        const double pred = h(model->weights, sample->features, data->num_features);

        const double error = pred - sample->label;
        for (size_t j = 0; j < data->num_features; ++j)
        {
            grad.values[j] += error * sample->features[j];
        }
    }

    for (size_t j = 0; j < data->num_features; ++j)
    {
        grad.values[j] *= 2.0 / data->num_samples;
    }
    return grad;
}

double mse_error(const Data *data, const Model *model)
{
    double loss = 0.0;
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        const double pred = h(model->weights, sample->features, data->num_features);

        const double error = pred - sample->label;
        loss += error * error;
    }
    return loss / data->num_samples;
}
