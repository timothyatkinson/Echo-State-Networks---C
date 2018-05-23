#include "esn.h"
#include "train.h"
#include "included_datasets.h"

int
main (void)
{
  srand(time(NULL));

    printf("Generating dataset\n");
    train_dataset* dataset = NARMA__10_dataset(6000, 2000, 2000, 100, 0.3, 0.05, 0.1, 1.0, 0.0, 0.5);
    printf("Training\n");
    double betas[5];
    betas[0] = 0.1;
    betas[1] = 0.001;
    betas[2] = 0.00001;
    betas[3] = 0.0000001;
    betas[4] = 0.000000001;

    int runs = 100;

    double train_scores[runs];
    double best_train = 999999;
    double validate_scores[runs];
    double best_validate = 999999;
    double test_scores[runs];
    double best_test = 999999;

    double best_lr;
    double best_is;
    double best_sr;
    double best_s;

    int nodes = 200;

    for(int i = 0; i < runs; i++){
      double leak_rate = rand_range(0.0, 1.0);
      double input_scale = rand_range(-1.0, 1.0);
      double spectral_radius = rand_range(-1.0, 1.0);
      double sparsity = rand_range(0.005, 1.0);
      ESN* esn = empty_esn(1, 1, nodes, leak_rate, input_scale, spectral_radius);
      printf("ESN %d: sparsity = %lf | leak rate = %lf | input scale = %lf | spectral radius = %lf\n", i, sparsity, leak_rate, input_scale, spectral_radius);
      randomize_esn(esn, sparsity);
      train_esn_ridge_regression(esn, dataset, 0, 1, betas, 5);
      double train_score = nmse(esn, dataset, 0);
      double validate_score = nmse(esn, dataset, 1);
      double test_score = nmse(esn, dataset, 2);
      train_scores[i] = train_score;
      validate_scores[i] = validate_score;
      test_scores[i] = test_score;
      if(train_score < best_train){
        best_train = train_score;
      }
      if(validate_score < best_validate){
        best_validate = validate_score;
        best_lr = leak_rate;
        best_is = input_scale;
        best_sr = spectral_radius;
        best_s = sparsity;
      }
      if(test_score < best_test){
        best_test = test_score;
      }
      printf("  scores: %lf | %lf | %lf\n", train_score, validate_score, test_score);
      free_esn(esn);

    }

    printf("Train: mean %lf best %lf\n", train_mean(train_scores, runs), best_train);
    printf("Validate: mean %lf best %lf\n", train_mean(validate_scores, runs), best_validate);
    printf("Test: mean %lf best %lf\n", train_mean(test_scores, runs), best_test);

    printf("\n\nBest validate params: [%lf, %lf, %lf, %lf]\n", best_s, best_lr, best_is, best_sr);
    ESN* esn = empty_esn(1, 1, nodes, best_lr, best_is, best_sr);
    for(int i = 0; i < 20; i++){
      randomize_esn(esn, best_s);
      train_esn_ridge_regression(esn, dataset, 0, 1, betas, 5);
      double train_score = nmse(esn, dataset, 0);
      double validate_score = nmse(esn, dataset, 1);
      double test_score = nmse(esn, dataset, 2);
      printf("  scores: %lf | %lf | %lf\n", train_score, validate_score, test_score);
    }

    free(esn);

    train_dataset_free(dataset);



  return 0;
}

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
