#include "train.h"

train_table* get_table(train_dataset* dataset, const int type);


train_table* get_table(train_dataset* dataset, const int type){
  if(type == TRAIN_CONST){
    return dataset->train;
  }
  if(type == VALIDATE_CONST){
    return dataset->validate;
  }
  if(type == TEST_CONST){
    return dataset->test;
  }
  return NULL;
}
/**train_get_X - TRAIN GET X
  *Gets the Matrix X for a given ESN and table. X is the matrix formed by [1, uN, state] for each input.
    *esn. The ESN to produce X for.
    *table. The table to produce X from.
*/
gsl_matrix* train_get_X(ESN* esn, train_table* table){
  gsl_matrix* X = gsl_matrix_alloc(1 + esn->inputs + esn->nodes, table->entries);
  for(int i = 0; i < table->warmups; i++){
    update_esn(esn, table->warmup_m);
  }
  for(int i = 0; i < table->entries; i++){
    update_esn(esn, table->uN[i]);
    for(int j = 0; j < esn->inputs + 1; j++){
      gsl_matrix_set(X, j, i, gsl_matrix_get(table->uN[i], j, 0));
    }
    for(int j = 0; j < esn->nodes; j++){
      gsl_matrix_set(X, j + 1 + esn->inputs, i, gsl_matrix_get(esn->state, j, 0));
    }
  }
  return X;
}

/**train_esn_pinverse - TRAIN ESN PSEUDOINVERSE
  *Trains an ESN using the pinverse method.
  * Wout = y_target . pinverse(X).
    *esn. The esn to train. state is reset to zeros at start and end.
    *dataset. The dataset to train against.
    *type. The table of the dataset to use. Typically 0 (train).
*/
void train_esn_pinverse(ESN* esn, train_dataset* dataset, const int type){
  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }

  train_table* table = get_table(dataset, type);

  gsl_matrix* X = train_get_X(esn, table);

  gsl_matrix* xInv = gsl_matrix_pinv(X, 0.0000001);

  gsl_matrix* y_target = gsl_matrix_alloc(esn->outputs, table->entries);

  for(int i = 0; i < table->entries; i++){
    gsl_matrix_set(y_target, 0, i, table->y_target[i]);
  }

  gsl_matrix_free(esn->wOut);
  esn->wOut = gsl_matrix_multiply(y_target, xInv);

  gsl_matrix_free(X);
  gsl_matrix_free(xInv);
  gsl_matrix_free(y_target);

  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }
}

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
void train_esn_ridge_regression(ESN* esn, train_dataset* dataset, const int train_type, const int beta_type, double* betas, int beta_count){

  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }

  train_table* table = get_table(dataset, train_type);

  gsl_matrix* X = train_get_X(esn, table);

  gsl_matrix* XXt = gsl_matrix_multiply_transpose_b(X, X);

  gsl_matrix* y_target = gsl_matrix_alloc(esn->outputs, table->entries);

  gsl_matrix* id = gsl_matrix_alloc(XXt->size1, XXt->size2);

  gsl_matrix_set_identity(id);

  for(int i = 0; i < table->entries; i++){
    gsl_matrix_set(y_target, 0, i, table->y_target[i]);
  }

  gsl_matrix* y_Xt = gsl_matrix_multiply_transpose_b(y_target, X);

  gsl_matrix* best_wOut = esn->wOut;
  double best_score = 99999999999999.9;

  for(int i = 0; i < beta_count; i++){
    double beta = betas[i];

    gsl_matrix* beta_id = gsl_matrix_alloc(XXt->size1, XXt->size2);
    gsl_matrix_memcpy(beta_id, id);
    gsl_matrix_scale(beta_id, beta);

    gsl_matrix_add(beta_id, XXt);
    gsl_matrix* inverse = gsl_matrix_inverse(beta_id);
    gsl_matrix* w_candidate = gsl_matrix_multiply(y_Xt, inverse);

    gsl_matrix_free(inverse);

    esn->wOut = w_candidate;

    double nmse_new = nmse(esn, dataset, beta_type);
    if(nmse_new < best_score){
      best_score = nmse_new;
      gsl_matrix_free(best_wOut);
      best_wOut = w_candidate;
    }
    else{
      gsl_matrix_free(w_candidate);
    }

    gsl_matrix_free(beta_id);
  }

  esn->wOut = best_wOut;

  gsl_matrix_free(X);
  gsl_matrix_free(XXt);
  gsl_matrix_free(y_target);
  gsl_matrix_free(y_Xt);
  gsl_matrix_free(id);


  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }
}

/**train_print - TRAIN PRINT
  *prints the behaviour of an ESN on a given dataset.
    *esn. The esn to run.
    *dataset. The dataset to use.
    *type. The table of the dataset to use..
*/
void train_print(ESN* esn, train_dataset* dataset, const int type){
  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }

  train_table* table = get_table(dataset, type);

  gsl_matrix* X = train_get_X(esn, table);

  gsl_matrix* Y = gsl_matrix_multiply(esn->wOut, X);

  for(int i = 0; i < table->entries; i++){
    printf("IN:\t");
    for(int j = 0; j < esn->inputs + 1; j++){
		  printf("%f\t", gsl_matrix_get(table->uN[i], j, 0));
    }
    printf("Expected:\t");
		printf("%f\t", table->y_target[i]);
    printf("OUT:\t");
		printf("%f\t\n", gsl_matrix_get(Y, 0, i));
  }

  gsl_matrix_free(X);
  gsl_matrix_free(Y);

  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }
}
/**train_table_free - TRAIN TABLE FREE
  *Frees a train_table, including warmup_m and uN.
    *table. The table to free.
*/
void train_table_free(train_table* table){
  for(int i = 0; i < table->entries; i++){
    gsl_matrix_free(table->uN[i]);
  }
  free(table->uN);
  gsl_matrix_free(table->warmup_m);
  free(table->y_target);
  free(table);
}

/**train_dataset_free - TRAIN DATASET FREE
  *Frees a train_dataset by freeing each of its train_tables using train_table_free.
    *dataset. The dataset to free.
*/
void train_dataset_free(train_dataset* dataset){
  train_table_free(dataset->train);
  train_table_free(dataset->validate);
  train_table_free(dataset->test);
  free(dataset);
}

double train_mean(double* vals, int count){
  double sum = 0.0;
  for(int i = 0; i < count; i++){
    sum += vals[i];
  }
  return sum / (double)count;
}

double train_variance(double* vals, int count){
  double mean = train_mean(vals, count);
  double sqDiff = 0.0;
  for(int i = 0; i < count; i++){
    sqDiff += (fabs(vals[i] - mean) * fabs(vals[i] - mean));
  }
  return sqDiff / (double)count;
}

/**nmse - NMSE
  *Computes the NMSE = 1/n * sum of 1 to n of (y_target[i] - y_actual[i])^2 / variance(y_target)
    *esn. The ESN to compute the NMSE using.
    *dataset. The dataset to compute the NMSE using.
    *type. The table of the dataset to use. Typically 2 (test).
*/
double nmse(ESN* esn, train_dataset* dataset, const int type){


  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(esn->state, i, 0, 0.0);
  }

  double sum = 0.0;
  train_table* table = get_table(dataset, type);
  int entries = table->entries;

  gsl_matrix* X = train_get_X(esn, table);

  gsl_matrix* Y = gsl_matrix_multiply(esn->wOut, X);

  double v = train_variance(table->y_target, entries);

  for(int i = 0; i < entries; i++){
    sum += ((table->y_target[i] - gsl_matrix_get(Y, 0, i)) * (table->y_target[i] - gsl_matrix_get(Y, 0, i)) / v);
  }

  gsl_matrix_free(X);
  gsl_matrix_free(Y);

  return sum / (double)entries;
}
