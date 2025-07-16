//
// Created by escha on 16.07.25.
//

#include <stdio.h>
#include <stdlib.h>

#include "ml.h"

int get_data_from_file(const char *filename, Data *data)
{
    FILE *file;
    if ((file = fopen(filename, "r")) == NULL) return 1;
    char *line = NULL;
    size_t line_len = 0;
    ssize_t x = 0;
    ssize_t y = 0;
    while (getline(&line, &line_len, file) != -1 && x < MAX_SAMPLES)
    {
        double impendace = 0;
        double vs = 0;

        if (sscanf(line, "%lf %lf\n", &impendace, &vs) < 2) return -1;
        data->x[x++] = impendace;
        data->y[y++] = vs;
    }
    data->num_samples = x;
    free(line);
    return 0;
}

double compute_loss(const Data *data, const Model *model)
{
    double loss = 0.0;
    for (size_t i = 0; i < data->num_samples; ++i)
    {
        const double pred = model->weights[1] * data->x[i] + model->weights[0];
        const double error = pred - data->y[i];
        loss += error * error;
    }
    return loss / data->num_samples;
}

void learn(const Data *data, Model *model, Gradient (*grad_loss)(const Data *, const Model *))
{
    for (int i = 0; i < 100000; ++i)
    {
        const Gradient grad = grad_loss(data, model);
        model->weights[0] -= LEARNING_RATE * grad.values[0];
        model->weights[1] -= LEARNING_RATE * grad.values[1];
        if (i % 1000 == 0) {
            const double loss = compute_loss(data, model);
            printf("Iteration %d: loss = %f\n", i, loss);
        }
    }
}