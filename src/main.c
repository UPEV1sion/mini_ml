#include <stdio.h>

#include "classification/gradients.h"
#include "ml/ml.h"
#include "regression/gradients.h"


int main(void)
{
    Sample samples[MAX_SAMPLES];
    Data data;
    data.capacity = MAX_SAMPLES;
    data.samples = samples;
    Model model = {0};
    get_data_from_file("/home/escha/CLionProjects/ai/assets/classification.csv", &data);
    learn(&data, &model, log_loss_gradient, zero_one_loss);

    printf("Weights: ");
    for (int i = 0; i < data.num_features; ++i)
    {
        printf("%lf ", model.weights[i]);
    }
    return 0;
}
