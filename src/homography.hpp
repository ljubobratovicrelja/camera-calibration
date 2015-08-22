#ifndef HOMOGRAPHY_HPP_ERDRJZXL
#define HOMOGRAPHY_HPP_ERDRJZXL


#include <matrix.hpp>
#include <vector.hpp>
#include <matfunc.hpp>
#include <optimization.hpp>


enum H_calc_alg {
	HOMOGRAPHY_8_POINT, //!< 8-point algorithm.
	HOMOGRAPHY_LEAST_SQUARES, //!< Least squares homography solver.
	HOMOGRAPHY_DLT //!< Normalized direct linear transformation homography solver.
};

/*
 * @brief Solve homography using non-normalized 8-point algorithm.
 *
 * Solve homography transform matrix which relates image points with world points, using non-normalized 8-point algorithm.
 *
 * m - image 2D point
 * M - world 3D point
 * H - homography matrix
 *
 * m = H*M
 *
 * @param image_points 2D image points.
 * @param model_points 3D world points.
 * @param alg Algorithm used to estimate homography.
 *
 * @return
 * cv::matrixr 3x3 homography matrix.
 */
cv::matrixr homography_solve(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, H_calc_alg alg = HOMOGRAPHY_8_POINT);

//! Evaluate optmization for given data set with given function.
int homography_optimize(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points,
                        cv::matrixr &H, real_t tol = 1e-14);

/*!
 * @brief Calculate reprojection error for homography calculated using given source and target points.
 *
 * @param H Homography 3x3 matrix
 * @param source_pts 2D source image points used to compute given homography.
 * @param target_pts 3D target world points used to compute given homography.
 *
 * @return 
 * Mean distance error from reprojected points using given homogrphy matrix.
 */
real_t calc_h_reprojection_error(const cv::matrixr &H, const std::vector<cv::vec2r> &source_pts, const std::vector<cv::vec3r> &target_pts);

#endif /* end of include guard: HOMOGRAPHY_HPP_ERDRJZXL */

