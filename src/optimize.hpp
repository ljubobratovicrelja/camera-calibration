#ifndef OPTIMIZE_HPP_OBWEIHS0
#define OPTIMIZE_HPP_OBWEIHS0

#include <optimization.hpp>

#include "calib.hpp"

int optimize_extrinsics(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, cv::matrixr &K, double tol = 1e-14);


int optimize_distortion(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, const std::vector<cv::matrixr> &K, cv::vec2r &k, double tol = 1e-14);

int optimize_all(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, std::vector<cv::matrixr> &K, cv::vec2r &k, bool fixed_aspect = false, bool no_skew = false, double tol = 1e-14);

#endif /* end of include guard: OPTIMIZE_HPP_OBWEIHS0 */
