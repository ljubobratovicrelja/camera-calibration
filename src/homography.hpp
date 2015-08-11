// The MIT License (MIT)
//
// Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Description:
// Homography calculation module.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


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
                        cv::matrixr &H, double tol = 10e-32);

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

