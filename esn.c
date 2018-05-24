#include "esn.h"
#include "train.h"
#include "included_datasets.h"

/**empty_esn - EMPTY ESN
  * Generates an ESN. All matrices are zero'd.
    * inputs. The number of inputs the ESN will handle.
    * outputs. The number of outputs the ESN will handle.
    * nodes. The number of nodes the ESN resevoir has.
    * leak_rate. The leak rate of the ESN.
    * input_scale. The input scaling of the ESN.
    * spectral_radius. The spectral radius of the ESN.
*/
ESN* empty_esn(int inputs, int outputs, int nodes, double leak_rate, double input_scale, double spectral_radius){
  ESN* esn = malloc(sizeof(ESN));
  esn->inputs = inputs;
  esn->outputs = outputs;
  esn->nodes = nodes;
  esn->leak_rate = leak_rate;
  esn->input_scale = input_scale;
  esn->spectral_radius = spectral_radius;
  esn->wIn = gsl_matrix_calloc(nodes, inputs + 1);
  esn->w = gsl_matrix_calloc(nodes, nodes);
  esn->wOut = gsl_matrix_calloc(outputs, (1 + nodes + inputs));
  esn->state = gsl_matrix_calloc(nodes, 1);
}

/**print_esn - PRINT ESN
  * Prints an ESN by printing the ESN's current (vector) state. ARGS:
    * esn - The ESN to print.
*/
void print_esn(ESN* esn){
  printf("\nESN with %d inputs, %d nodes, %d outputs, leak rate %lf, input scaling %lf.\n", esn->inputs, esn->nodes, esn->outputs, esn->leak_rate, esn->input_scale);
  printf("\n Current state \n\n");
  print_matrix(esn->state);
}

/**print_esn_full - PRINT ESN FULL
  * Prints an ESN by printing each weight matrix and the ESN's current (vector) state. ARGS:
    * esn - The ESN to print.
*/
void print_esn_full(ESN* esn){
  printf("\nESN with %d inputs, %d nodes, %d outputs, leak rate %lf, input scaling %lf.\n", esn->inputs, esn->nodes, esn->outputs, esn->leak_rate, esn->input_scale);
  printf("\n wIn \n\n");
  print_matrix(esn->wIn);
  printf("\n w \n\n");
  print_matrix(esn->w);
  printf("\n wOut \n\n");
  print_matrix(esn->wOut);
  printf("\n Current state \n\n");
  print_matrix(esn->state);
}

/**free_esn - FREE ESN
  * Frees an ESN including its various gsl_matrix weights.
    * esn - The ESN to free.
*/
void free_esn(ESN* esn){
  gsl_matrix_free(esn->wIn);
  gsl_matrix_free(esn->w);
  gsl_matrix_free(esn->wOut);
  gsl_matrix_free(esn->state);
  free(esn);
}

/**update_esn - UPDATE ESN
  * Steps an ESN along according to input uN.
  * Update x(t) = tanh((wIn * input_scale).uN + (w * spectral_radius).x'(t - 1))
  * New state x'(t) = (1 - leak_rate) * x'(t - 1) + leak_rate * x(t)
  * x'(t - 1) is the old state at time t
    * esn. The ESN to update.
    * uN. The inputs to the ESN to update. uN is assumed to be prefaced with the bias e.g. [1; inputs]
*/
void update_esn(ESN* esn, gsl_matrix* uN){
  gsl_matrix* new_state = gsl_matrix_multiply(esn->wIn, uN);
  gsl_matrix_scale(new_state, esn->input_scale);
  gsl_matrix* res_state = gsl_matrix_multiply(esn->w, esn->state);
  gsl_matrix_scale(res_state, esn->spectral_radius);
  gsl_matrix_add(new_state, res_state);
  for(int i = 0; i < esn->nodes; i++){
    gsl_matrix_set(new_state, i, 0, tanh(gsl_matrix_get(new_state, i, 0)));
  }
  gsl_matrix_scale(esn->state, 1.0 - esn->leak_rate);
  gsl_matrix_scale(new_state, esn->leak_rate);
  gsl_matrix_add(new_state, esn->state);
  gsl_matrix_free(esn->state);
  gsl_matrix_free(res_state);
  esn->state = new_state;
}

/**randomize_esn - RANDOMIZE ESN
  * Randomizes an ESN's weight matrices. Input weights (wIn) are uniformally chosen from the interval [-1, 1]. Resevoir weights (w) occur with probability (density) and
  * are uniformally chosen from the interval [-0.5, 0.5]. The resevoir weights (w) are then scaled by their (1 / maximum eigenvalue).
    * esn. The esn to randomize
    * density. How sparse the esn should be.
*/
void randomize_esn(ESN* esn, double density){
  for(int i = 0; i < esn->nodes; i++){
    for(int j = 0; j < esn->inputs + 1; j++){
      gsl_matrix_set(esn->wIn, i, j, rand_range(-1.0, 1.0));
    }
  }
  bool first = true;
  if(density != 0.0){
    while(first || gsl_matrix_max_eigenvalue(esn->w) == 0.0){
      first = false;
      for(int i = 0; i < esn->nodes; i++){
        for(int j = 0; j < esn->nodes; j++){
          if(rand_bool(density)){
            gsl_matrix_set(esn->w, i, j, rand_range(-0.5, 0.5));
          }
          else{
            gsl_matrix_set(esn->w, i, j, 0);
          }
        }
      }
    }
    gsl_matrix_scale(esn->w, 1.0 / gsl_matrix_max_eigenvalue(esn->w));
  }
}
