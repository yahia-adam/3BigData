// mlp_model.h
#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
    extern "C" {
        #endif
            typedef struct MultiLayerPerceptron MultiLayerPerceptron;
            MultiLayerPerceptron *init_mlp(size_t *npl, size_t npl_size);
            void train_mlp(MultiLayerPerceptron *model);
            float *predict_mlp(MultiLayerPerceptron *model);
        #ifdef __cplusplus
    }
#endif

#endif // MLP_MODEL_H