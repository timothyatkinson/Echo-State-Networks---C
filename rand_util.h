#ifndef RU_H
#define RU_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**rand_double - RAND DOUBLE
  *Computes a random double between 0 and 1 using C's inbuilt RNG
*/
double rand_double();

/**rand_double - RAND DOUBLE
  *Computes a random double between min and max (inclusive ish) using C's inbuilt RNG
    * min. The minimum value
    * max. The maximum value
*/
double rand_range(double min, double max);

/**rand_bool - RAND BOOLEAN
  * Computes a random boolean which is true with probability p.
  * p. The probability of returning true
*/
bool rand_bool(double p);

#endif
