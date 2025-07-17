//
// Created by escha on 16.07.25.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ml.h"

#define MAX_LINE_LEN 1024

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

void min_max_scale(Data *data)
{
    for (int i = 1; i < data->num_features; ++i) // ignore bias
    {
        double min_val = data->samples[0].features[i];
        double max_val = data->samples[0].features[i];

        for (int j = 1; j < data->num_samples; ++j)
        {
            const double val = data->samples[j].features[i];
            min_val = MIN(min_val, val);
            max_val = MAX(max_val, val);
        }


        data->feature_min[i] = min_val;
        data->feature_max[i] = max_val;

        double range = max_val - min_val;
        if (range == 0) range = 1.0;

        for (int j = 0; j < data->num_samples; ++j)
        {
            data->samples[j].features[i] = (data->samples[j].features[i] - min_val) / range;
        }
    }
}

void learn(
    Data *data,
    Model *model,
    Gradient (*grad_loss)(const Data *, const Model *),
    double (*error)(const Data *, const Model *))
{
    const double initial_lr = 0.0001;
    const double decay_rate = 1e-5;

    min_max_scale(data);

    double best_loss = 1e10;
    int no_improve = 0;
    int patience = 100;

    for (int i = 0; i < ITERATIONS; ++i)
    {
        const double learning_rate = initial_lr / (1.0 + decay_rate * i);

        Gradient grad = grad_loss(data, model);
        for (int j = 0; j < data->num_features; ++j)
            model->weights[j] -= learning_rate * grad.values[j];

        if (i % 10000 == 0)
        {
            const double loss = error(data, model);
            printf("Iteration %d: loss = %f\n", i, loss);

            if (loss < best_loss)
            {
                best_loss = loss;
                no_improve = 0;
            } else
            {
                no_improve++;
                if (no_improve >= patience)
                {
                    printf("Stopping early at iteration %d\n", i);
                    break;
                }
            }
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
