//
// Created by escha on 16.07.25.
//

#pragma once

#define LEARNING_RATE 0.000000000001
#define MAX_WEIGHTS 10
#define MAX_SAMPLES 10000

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
} Data;

typedef struct
{
    double values[MAX_WEIGHTS];
} Gradient;

int get_data_from_file(const char *filename, Data *data);
double compute_loss(const Data *data, const Model *model);
void learn(const Data *data, Model *model, Gradient (*grad_loss)(const Data *, const Model *));
