#include "included_datasets.h"
#include "rand_util.h"
#include <gsl/gsl_matrix.h>


/**NARMA_10_dataset - NARMA 10 DATASet
  *Generates a train_dataset according to the formula:
  y(t + 1) = d.(a.y(t) + b.y(t).(sum for j in 1 to 10 of y(t - j)) + 1.5x + c) were x is generated from [iMin, iMax]
    *train_entries. The number of training entries.
    *validation_entries. The number of validation entires.
    *test_entries. The number of test entries.
    *warmup. The number of warmup steps for the ESN.
    *a. The a component of the above formula.
    *b. The b component of the above formula.
    *c. The c component of the above formula.
    *d. The d component of the above formula.
    *iMin. The minimum value of x.
    *iMax. The maximum value of x.
*/
train_dataset* NARMA__10_dataset(int train_entries, int validation_entries, int test_entries, int warmup, double a, double b, double c, double d, double iMin, double iMax){
  train_dataset* dataset = malloc(sizeof(train_dataset));

  dataset->train = NARMA_10_table(train_entries, warmup, a, b, c, d, iMin, iMax);
  dataset->validate = NARMA_10_table(validation_entries, warmup, a, b, c, d, iMin, iMax);
  dataset->test = NARMA_10_table(test_entries, warmup, a, b, c, d, iMin, iMax);

  return dataset;
}

/**NARMA_10_dataset - NARMA 10 DATASet
  *Generates a train_table according to the formula:
  y(t + 1) = d.(a.y(t) + b.y(t).(sum for j in 1 to 10 of y(t - j)) + 1.5x + c) were x is generated from [iMin, iMax]
    *entries. The number of entries.
    *warmup. The number of warmup steps for the ESN.
    *a. The a component of the above formula.
    *b. The b component of the above formula.
    *c. The c component of the above formula.
    *d. The d component of the above formula.
    *iMin. The minimum value of x.
    *iMax. The maximum value of x.
*/
train_table* NARMA_10_table(int entries, int warmup, double a, double b, double c, double d, double iMin, double iMax){
  train_table* table = malloc(sizeof(train_table));

  table->entries = entries;
  table->warmups = warmup;

  gsl_matrix* warmup_m = gsl_matrix_alloc(2, 1);
  gsl_matrix_set(warmup_m, 0, 0, 1.0);
  gsl_matrix_set(warmup_m, 1, 0, 0.0);

  table->warmup_m = warmup_m;

  table->uN = malloc(entries * sizeof(gsl_matrix*));
  table->y_target = malloc(entries * sizeof(double));

  for(int i = 0; i < entries; i++){
    table->uN[i] = gsl_matrix_alloc(2, 1);
    double x = rand_range(iMin, iMax);
    gsl_matrix_set(table->uN[i], 0, 0, 1.0);
    gsl_matrix_set(table->uN[i], 1, 0, x);

    double x10 = 0.0;
    if(i - 10 >= 0){
      x10 = gsl_matrix_get(table->uN[i - 10], 1, 0);
    }

    double sum = 0.0;
    for(int j = 2; j <= 10; j++){
      if(i - j >= 0){
        sum += table->y_target[i - j];
      }
    }
    double last = 0.0;

    if(i != 0){
      last = table->y_target[i - 1];
    }

    table->y_target[i] = d * ((a * last) + (b * last * sum) + (1.5 * x * x10) + c);
  }

  return table;

}
