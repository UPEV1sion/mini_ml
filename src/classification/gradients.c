//
// Created by escha on 16.07.25.
//

#include <stddef.h>
#include <math.h>
#include "gradients.h"

// 1/m sum_{i=1}^{m} (-y_i * x_i) / (1 + exp(y_i <w, x_i>)
Gradient log_loss_gradient(const Data *data, const Model *model)
{
    Gradient grad = {0};
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        const double pred = h(model->weights, sample->features, data->num_features);

        const double y = sample->label;
        const double denom = 1 + exp(y * pred);
        for (size_t j = 0; j < data->num_features; ++j)
        {
            grad.values[j] += (-y * sample->features[j]) / denom;
        }
    }

    for (int j = 0; j < data->num_features; ++j)
    {
        grad.values[j] /= data->num_samples;
    }

    return grad;
}

double zero_one_loss(const Data *data, const Model *model)
{
    int errors = 0;
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const Sample *sample = data->samples + i;
        const double pred = h(model->weights, sample->features, data->num_features);

        const int predicted_label = (pred >= 0) ? 1 : -1;

        if (predicted_label != (int) (sample->label))
        {
            errors++;
        }
    }

    return (double)errors / data->num_samples;
}
