#include <stdio.h>

#include "ml/ml.h"
#include "regression/gradients.h"


int main(void)
{
    Data data;
    Model model = {0};
    get_data_from_file("/home/escha/CLionProjects/ai/assets/sand_dataset.csv", &data);
    learn(&data, &model, mse_gradient);


    printf("loss: %f, %f\n", model.weights[0], model.weights[1]);
    return 0;
}
