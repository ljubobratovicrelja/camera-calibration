#ifndef HOMOGRAPHY_HPP_ERDRJZXL
#define HOMOGRAPHY_HPP_ERDRJZXL


#include <matrix.hpp>
#include <vector.hpp>
#include <matfunc.hpp>
#include <optimization.hpp>


/*
 * @brief Solve homography using Direct Linear Transformation (DLT) algorithm.
 *
 * m - image 2D point
 * M - world 3D point
 * H - homography matrix
 *
 * m = H*M
 *
 * @param image_points 2D image points.
 * @param model_points 3D world points.
 *
 * @return
 * cv::matrixr 3x3 homography matrix.
 */
cv::matrixr homography_solve(const std::vector<cv::vec2r>& image_points, const std::vector<cv::vec3r>& model_points);

// ! Evaluate optmization for given data set with given function.
int homography_optimize(const std::vector<cv::vec2r>& image_points,
                        const std::vector<cv::vec3r>& model_points,
                        cv::matrixr& H,
                        real_t tol = 1e-14);

#endif /* end of include guard: HOMOGRAPHY_HPP_ERDRJZXL */
