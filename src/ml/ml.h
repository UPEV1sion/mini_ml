//
// Created by escha on 16.07.25.
//

#pragma once

#define LEARNING_RATE 0.0001
#define ITERATIONS 500000

#define MAX_WEIGHTS 10
#define MAX_SAMPLES 1000

typedef struct
{
    double weights[MAX_WEIGHTS];
} Model;

typedef struct
{
    double features[MAX_WEIGHTS];
    double label;
} Sample;

typedef struct
{
    Sample *samples;
    int num_samples;
    int capacity;
    int num_features;

    double feature_min[MAX_WEIGHTS];
    double feature_max[MAX_WEIGHTS];
} Data;

typedef struct
{
    double values[MAX_WEIGHTS];
} Gradient;

int get_data_from_file(const char *filename, Data *data);
void min_max_scale(Data *data);
void learn(Data *data, Model *model, Gradient (*grad_loss)(const Data *, const Model *), double (*error)(const Data *, const Model *));
double h(const double *w, const double *x, size_t n);
