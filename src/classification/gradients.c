//
// Created by escha on 16.07.25.
//

#include <stddef.h>
#include <math.h>
#include "gradients.h"

// (-y_i * x_i) / (1 + exp(y_i <w, x_i>)
Gradient log_loss_gradient(const Data *data, const Model *model)
{
    Gradient grad = {0};
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        double dot = 0.0;

        // dot product
        for (size_t j = 0; j < data->num_features; ++j)
        {
            dot += model->weights[j] * sample->features[j];
        }

        const double y = sample->label;
        double denom = 1 + exp(y * dot);
        for (size_t j = 0; j < data->num_features; ++j)
        {
            grad.values[j] += (-y * sample->features[j])/denom;
        }
    }

    for (int j = 0; j < data->num_features; ++j) {
        grad.values[j] /= data->num_samples;
    }

    return grad;
}
