#ifndef TRAIN_H
#define TRAIN_H

#include <gsl/gsl_matrix.h>
#include "esn.h"
#include "matrix_util.h"

static const int TRAIN_CONST = 0;
static const int VALIDATE_CONST = 1;
static const int TEST_CONST = 2;

/** STRUCT train_table - TRAIN TABLE
  *A single table - train, validate or test - for a train_dataset.
    *entries. How many rows the dataset has.
    *warmups. How many times an ESN should be run on warmup_m before being run on the dataset.
    *warmup_m. The warmup input (typically zeros).
    *uN. The inputs.
    *y_target. The outputs.
*/
typedef struct train_table{
  int entries;
  int warmups;
  gsl_matrix* warmup_m;
  gsl_matrix** uN;
  double* y_target;
} train_table;

/** STRUCT train_dataset - TRAIN DATASET
  *A dataset containing a training table, a validation table and a testing table.
    *train. The training table.
    *validate. The validation table.
    *test. The testing table.
*/
typedef struct train_dataset{
  train_table* train;
  train_table* validate;
  train_table* test;
} train_dataset;


double train_mean(double* vals, int count);

double train_variance(double* vals, int count);

/**nmse - NMSE
  *Computes the NMSE = 1/n * sum of 1 to n of (y_target[i] - y_actual[i])^2 / variance(y_target)
    *esn. The ESN to compute the NMSE using.
    *dataset. The dataset to compute the NMSE using.
    *type. The table of the dataset to use. Typically 2 (test).
*/
double nmse(ESN* esn, train_dataset* dataset, const int type);

/**train_print - TRAIN PRINT
  *prints the behaviour of an ESN on a given dataset.
    *esn. The esn to run.
    *dataset. The dataset to use.
    *type. The table of the dataset to use..
*/
void train_print(ESN* esn, train_dataset* dataset, const int type);

/**train_get_X - TRAIN GET X
  *Gets the Matrix X for a given ESN and table. X is the matrix formed by [1, uN, state] for each input.
    *esn. The ESN to produce X for.
    *table. The table to produce X from.
*/
gsl_matrix* train_get_X(ESN* esn, train_table* table);

/**train_esn_pinverse - TRAIN ESN PSEUDOINVERSsE
  *Trains an ESN using the pinverse method.
  * Wout = y_target . pinverse(X). X is collected from the train_get_X method.
    *esn. The esn to train. state is reset to zeros at start and end.
    *dataset. The dataset to train against.
    *type. The table of the dataset to use. Typically 0 (train).
*/
void train_esn_pinverse(ESN* esn, train_dataset* dataset, const int type);


/**train_esn_ridge_regression - TRAIN ESN RIDGE REGRESSION
  *Trains an ESN using the ridge regression method. This is cheaper than pinverse but not guaranteed to find a global optimum
  * Wout = y_target . Xt . inv(XXt + betaI).
    *esn. The esn to train. state is reset to zeros at start and end.
    *dataset. The dataset to train against.
    *train_type. The table of the dataset to use for training. Typically 0 (train).
    *beta_type. he table of the dataset to use for validating different beta values. Typically 1 (validate).
    *betas. The set of beta parameters to use. Each is used for training and the one that maximises the beta_type table's NMSE is the final one used.
    *beta_count. The number of beta parameters.
*/
void train_esn_ridge_regression(ESN* esn, train_dataset* dataset, const int train_type, const int beta_type, double* betas, int beta_count);

/**train_table_free - TRAIN TABLE FREE
  *Frees a train_table, including warmup_m and uN.
    *table. The table to free.
*/
void train_table_free(train_table* table);

/**train_dataset_free - TRAIN DATASET FREE
  *Frees a train_dataset by freeing each of its train_tables using train_table_free.
    *dataset. The dataset to free.
*/
void train_dataset_free(train_dataset* dataset);

#endif
