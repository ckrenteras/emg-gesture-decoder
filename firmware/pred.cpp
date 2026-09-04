#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <json.hpp>
#include <map>
#include "pred.hpp"

MatrixXd load_binary_matrix(const std::string& filepath, int rows, 
    int cols) {
    std::ifstream file(filepath, std::ios::binary);
    MatrixXd mat(rows, cols);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(float));
    }
    return mat;
}

void load_priors(const std::string& priors_path, lda_model& model) {
    std::ifstream p_file(priors_path);
    json p_json;
    p_file >> p_json;

    for (int id : model.class_labels) {
        std::string str_id = std::to_string(id);

        model.priors.push_back(p_json[str_id].get<double>());
    }
}

std::map<int, std::string> load_class_mappings(const std::string& filepath) {
    std::ifstream file(filepath);
    json j;
    file >> j;
    std::map<int, std::string> mappings;
    for (auto& [key, value] : j.items()) {
        mappings[std::stoi(key)] = value.get<std::string>();
    }
    return mappings;
}


int predict(const VectorXd& x, const lda_model& model) {
    int num_classes = model.num_classes;
    float max_score = -std::numeric_limits<float>::infinity();
    int best_class = 0;

    for (int i = 0; i < num_classes; ++i) {
        VectorXd diff = x - model.centroids.col(i);

        float dist_term = -0.5 * (diff.transpose() * model.inv_cov * diff).value();

        float score = std::log(model.priors[i]) + dist_term;

        if (score > max_score) {
            max_score = score;
            best_class = model.class_labels[i];
        }
    }
    return best_class;
}

void load_lda_model(lda_model& model, const std::string& priors_path,
    MatrixXd centroids, const std::string& mappings_path,
    const std::string& inv_cov_path, int cov_rows, int cov_cols) {
        std::map<int, std::string> mappings = load_class_mappings(mappings_path);
        int num_classes = static_cast<int>(mappings.size());
        std::vector<int> keys;
        keys.reserve(num_classes);
        for (const auto& [key, value] : mappings) {
            keys.push_back(key);
        }

        model.num_classes = num_classes;
        model.class_labels = keys;

        load_priors(priors_path, model);

        model.inv_cov = load_binary_matrix(inv_cov_path, cov_rows, cov_cols);
        model.centroids = centroids;
    }