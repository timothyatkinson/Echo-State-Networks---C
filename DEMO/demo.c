#include "../train.h"
#include "../esn.h"
#include "../included_datasets.h"

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

    int runs = 10000;

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
