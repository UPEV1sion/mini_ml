//
// Created by escha on 16.07.25.
//

#include <stdio.h>
#include <stdlib.h>

#include "ml.h"

#include <string.h>

#define MAX_LINE_LEN 1024

int get_data_from_file(const char *filename, Data *data)
{
    FILE *file;
    if ((file = fopen(filename, "r")) == NULL) return 1;

    char line[MAX_LINE_LEN];
    int count = 0;
    int num_features = -1;

    while (fgets(line, sizeof line, file) && count < MAX_SAMPLES)
    {
        Sample *sample = data->samples + count;

        char *tokens[MAX_WEIGHTS + 1];
        int token_count = 0;

        char *save_ptr;
        char *token = strtok_r(line, " \n\t", &save_ptr);
        while (token && token_count < MAX_WEIGHTS)
        {
            tokens[token_count++] = token;
            token = strtok_r(NULL, " \n\t", &save_ptr);
        }

        if (token_count < 2) return -2; // at least a label and a feature

        char *endptr;
        sample->label = strtod(tokens[token_count - 1], &endptr);
        if (*endptr != '\0') return -3;

        int column_index = 0;
        for (; column_index < token_count - 1; ++column_index)
        {
            sample->features[column_index] = strtod(tokens[column_index], &endptr);
            if (*endptr != '\0') return -4;
        }

        if (num_features == -1)
            num_features = column_index;
        else if (column_index != num_features)
            return -5;

        count++;
    }

    fclose(file);
    data->num_samples = count;
    data->num_features = num_features;

    return 0;
}

void learn(
    const Data *data,
    Model *model,
    Gradient (*grad_loss)(const Data *, const Model *),
    double (*error)(const Data *, const Model *))
{
    for (size_t i = 0; i < ITERATIONS; ++i)
    {
        const Gradient grad = grad_loss(data, model);
        for (size_t j = 0; j < data->num_features; ++j)
        {
            model->weights[j] -= LEARNING_RATE * grad.values[j];
        }
        if (i % 1000 == 0)
        {
            const double loss = error(data, model);
            printf("Iteration %zu: loss = %f\n", i, loss);
        }
    }
}

double h(const double *w, const double *x, const size_t n)
{
    double pred = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        pred += w[i] * x[i];
    }
    return pred;
}
