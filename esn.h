#ifndef ESN_H
#define ESN_H

#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "matrix_util.h"
#include "rand_util.h"
#include <time.h>

/**STRUCT ESN
 * The ESN struct stores all of the information required to run an ESN. The components are:
  * inputs - The number of inputs the ESN has.
  * outputs - The number of outputs the ESN has.
  * nodes - The number of nodes in the ESN's resevoir.
  * wIn - A [nodes x (inputs + 1)] GSL Matrix describing the weights between the ESN's resevoir nodes and the ESN's inputs. The first weight is the node's bias.
  * w - A [nodes x nodes] GSL Matrix describing the weights between the ESN's resevoir nodes.
  * wOut - A [outputs x (inputs + nodes + 1)] GSL Matrix describing the weights between the ESN's outputs and all other nodes. The first wieght is the output's bias,
    The next #input weights are the weights for the inputs and the final #nodes weights are the weights for the resevoir nodes.
  * leak_rate - The ESN's leak rate for updates.
  * input_scale - The ESN's input scaling for updates.
  * spectral_radius - The spectral radius of the esn
  * state - A #nodes long vector ([#nodes x 1] gsl_matrix) describing the current state of every node in the ESN resevoir.
*/
typedef struct ESN{
  int inputs;
  int outputs;
  int nodes;
  gsl_matrix* wIn;
  gsl_matrix* w;
  gsl_matrix* wOut;
  double leak_rate;
  double input_scale;
  double spectral_radius;
  gsl_matrix* state;
} ESN;

/**PRINT ESN
  * Prints an ESN by printing the ESN's current (vector) state. ARGS:
    * esn - The ESN to print.
*/
void print_esn(ESN* esn);

/**PRINT ESN FULL
  * Prints an ESN by printing each weight matrix and the ESN's current (vector) state. ARGS:
    * esn - The ESN to print.
*/
void print_esn_full(ESN* esn);

/**empty_esn - EMPTY ESN
  * Generates an ESN. All matrices are zero'd.
    * inputs. The number of inputs the ESN will handle.
    * outputs. The number of outputs the ESN will handle.
    * nodes. The number of nodes the ESN resevoir has.
    * leak_rate. The leak rate of the ESN.
    * input_scale. The input scaling of the ESN.
    * spectral_radius. The spectral radius of the ESN.
*/
ESN* empty_esn(int inputs, int outputs, int nodes, double leak_rate, double input_scale, double spectral_radius);

/**update_esn - UPDATE ESN
  * Steps an ESN along according to input uN.
  * Update x(t) = tanh((wIn * input_scale).uN + w.x'(t - 1))
  * New state x'(t) = (1 - leak_rate) * x'(t - 1) + leak_rate * x(t)
  * x'(t - 1) is the old state at time t
    * esn. The ESN to update.
    * uN. The inputs to the ESN to update. uN is assumed to be prefaced with the bias e.g. [1; inputs]
*/
void update_esn(ESN* esn, gsl_matrix* uN);

/**free_esn - FREE ESN
  * Frees an ESN including its various gsl_matrix weights.
    * esn - The ESN to free.
*/
void free_esn(ESN* esn);

/**randomize_esn - RANDOMIZE ESN_H
  * Randomizes an ESN's weight matrices. Input weights (wIn) are uniformally chosen from the interval [-1, 1]. Resevoir weights (w) occur with probability (density) and
  * are uniformally chosen from the interval [-0.5, 0.5].
    * esn. The esn to randomize
    * density. How sparse the esn should be.
*/
void randomize_esn(ESN* esn, double density);

#endif
