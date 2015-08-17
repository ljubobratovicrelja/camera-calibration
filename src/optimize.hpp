#ifndef OPTIMIZE_HPP_OBWEIHS0
#define OPTIMIZE_HPP_OBWEIHS0

#include <optimization.hpp>

#include "calib.hpp"

/*!
 * @brief Optimize extrinsic matrix K (3x4) by minimizing reprojection error while ignoring distortion.
 *
 * @param image_points Image 2D points from calibration pattern used to extract extrinsic matrix.
 * @param model_points World 3D points from calibration pattern.
 * @param A Intrinsic matrix 3x3.
 * @param K Initial extrinsic matrix.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_extrinsics(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, cv::matrixr &K, double tol = 1e-14);

/*!
 * @brief Optimize distortion parameters by minimizing reprojection.
 *
 * @param image_points Image 2D points from calibration patterns used for calibration.
 * @param model_points World 3D points from calibration pattern.
 * @param A Intrinsic matrix 3x3.
 * @param K Initial extrinsic matrices.
 * @param k Initial value for lens distortion parameters.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_distortion(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, const std::vector<cv::matrixr> &K, cv::vec2r &k, double tol = 1e-14);

/*!
 * @brief Optimize all calibration parameters.
 *
 * @param image_points Image 2D points from calibration patterns used for calibration.
 * @param model_points World 3D points from calibration pattern.
 * @param A Intrinsic matrix 3x3.
 * @param K Extrinsic matrices.
 * @param k Lens distortion parameters.
 * @param fixed_aspect Force fixed aspect ration (alpha = beta) in optimization.
 * @param no_skew Force zero skew (c) in optimization.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_all(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, std::vector<cv::matrixr> &K, cv::vec2r &k, bool fixed_aspect = false, bool no_skew = false, double tol = 1e-14);

#endif /* end of include guard: OPTIMIZE_HPP_OBWEIHS0 */
