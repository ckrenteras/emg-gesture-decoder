#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include <json.hpp>
#include <map>

inline const std::string PRIORS_PATH = "../models/v4/class_priors.json";
inline const std::string INV_COV_PATH = "../models/v4/lda_shared_covariance.npy";
inline const std::string MAPPINGS_PATH = "../models/v4/class_mappings.json";

using json = nlohmann::json;
using namespace Eigen;

struct lda_model {
    std::vector<int> class_labels;
    std::vector<float> priors;
    int num_classes;
    MatrixXd inv_cov;
    MatrixXd centroids;
};

MatrixXd load_binary_matrix(const std::string& filepath, int rows, int cols);

void load_priors(const std::string& priors_path, lda_model& model);

std::map<int, std::string> load_class_mappings(const std::string& filepath);

int predict(const VectorXd& x, const lda_model& model);

void load_lda_model(lda_model& model, const std::string& priors_path,
    MatrixXd centroids, const std::string& mappings_path,
    const std::string& inv_cov_path, int cov_rows, int cov_cols);