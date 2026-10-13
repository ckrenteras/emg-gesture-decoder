#include <vector>
#include <cmath>
#include <limits>
#include <ArduinoEigenDense.h>
#include "pred.hpp"
#include "feature_extraction.hpp"

static_assert(NUM_FEATURES == MODEL_NUM_FEATURES,
    "firmware feature count doesn't match the trained model - rerun export_model_header.py or fix feature_extraction");

int predict(const VectorXd& x, const lda_model& model) {
    int num_classes = model.num_classes;
    double max_score = -std::numeric_limits<double>::infinity();
    int best_class = 0;

    for (int i = 0; i < num_classes; ++i) {
        VectorXd diff = x - model.centroids.col(i);

        double dist_term = -0.5 * (diff.transpose() * model.inv_cov * diff).value();

        double score = std::log(model.priors[i]) + dist_term;

        if (score > max_score) {
            max_score = score;
            best_class = model.class_labels[i];
        }
    }
    return best_class;
}

void load_lda_model(lda_model& model, const MatrixXd& centroids) {
    model.num_classes = MODEL_NUM_CLASSES;
    model.class_labels.assign(MODEL_CLASS_IDS, MODEL_CLASS_IDS + MODEL_NUM_CLASSES);
    model.priors.assign(MODEL_PRIORS, MODEL_PRIORS + MODEL_NUM_CLASSES);
    model.inv_cov = Map<const Matrix<double, MODEL_NUM_FEATURES, MODEL_NUM_FEATURES, RowMajor>>(
        &MODEL_INV_COV[0][0]);
    model.centroids = centroids;
}
