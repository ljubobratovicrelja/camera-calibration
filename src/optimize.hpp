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
 * @param k Initial value for radial and tangential distortion parameters. As input can be 2, 4, and 8 dimension vector:
 *	2 - [k1, k2]
 *	4 - [k1, k2, p1, p2]
 *	8 - [k1, k2, k3, k4, k5, k6, p1, p2]
 * Returned value is always 8 point vector with all parameters. For every non-given parameter zero is set as initial value.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_distortion(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, const std::vector<cv::matrixr> &K, cv::vectorr&k, double tol = 1e-14);

/*!
 * @brief Optimize all calibration parameters for one pattern.
 *
 * @param image_points Image 2D points from calibration pattern.
 * @param model_points World 3D points from calibration pattern.
 * @param A Intrinsic matrix 3x3.
 * @param K Extrinsic matrix 3x4.
 * @param k Initial value for radial and tangential distortion parameters. As input can be 2, 4, and 8 dimension vector:
 *	2 - [k1, k2]
 *	4 - [k1, k2, p1, p2]
 *	8 - [k1, k2, k3, k4, k5, k6, p1, p2]
 * @param fixed_aspect Force fixed aspect ration (alpha = beta) in optimization.
 * @param no_skew Force zero skew (c) in optimization.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_all(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, cv::matrixr &K, cv::vectorr &k, bool fixed_aspect = false, bool no_skew = false, double tol = 1e-14);
/*!
 * @brief Optimize all calibration parameters.
 *
 * @param image_points Image 2D points from calibration patterns used for calibration.
 * @param model_points World 3D points from calibration pattern.
 * @param A Intrinsic matrix 3x3.
 * @param K Extrinsic matrices.
 * @param k Initial value for radial and tangential distortion parameters. As input can be 2, 4, and 8 dimension vector:
 *	2 - [k1, k2]
 *	4 - [k1, k2, p1, p2]
 *	8 - [k1, k2, k3, k4, k5, k6, p1, p2]
 * @param fixed_aspect Force fixed aspect ration (alpha = beta) in optimization.
 * @param no_skew Force zero skew (c) in optimization.
 * @param tol Error tolerance used in Levenberg-Marquard optimization algorithm.
 */ 
int optimize_all(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, std::vector<cv::matrixr> &K, cv::vectorr &k, bool fixed_aspect = false, bool no_skew = false, double tol = 1e-14);

#endif /* end of include guard: OPTIMIZE_HPP_OBWEIHS0 */
