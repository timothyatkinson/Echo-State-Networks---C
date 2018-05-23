#ifndef ID_H
#define ID_H
#include "train.h"

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
train_dataset* NARMA__10_dataset(int train_entries, int validation_entries, int test_entries, int warmup, double a, double b, double c, double d, double iMin, double iMax);

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
train_table* NARMA_10_table(int entries, int warmup, double a, double b, double c, double d, double iMin, double iMax);
#endif
