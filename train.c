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
