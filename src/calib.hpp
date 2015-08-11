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
// Collection of algoritms for calibration data computation.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef CALIB_HPP_Q5H9DDXY
#define CALIB_HPP_Q5H9DDXY

#include <image.hpp>
#include <improc.hpp>
#include <region.hpp>
#include <linalg.hpp>
#include <matfunc.hpp>

/*!
 * @brief Calculate object points for chessboard pattern used as calibration object.
 *
 * @param rows rows of corners present in the chessboard pattern.
 * @param cols columns of corners present in the chessboard pattern.
 * @param square_size real world size of the single chessboard square in generic units.
 *
 * @return
 * std::vector<cv::vec3r> world points of chessboard pattern with Z = 1, where X, Y is 2D homogenious point.
 */
std::vector<cv::vec3r> calculate_object_points(unsigned rows, unsigned cols, real_t square_size);

/*!
 * @brief Normalize image points used in calibration process.
 *
 * @param patterns vector of vector<cv::vec2r>, as in list of chessboard pattern image point corners.
 * @param w width of the image from which the pattern is extracted.
 * @param h height of the image from which the pattern is extracted.
 *
 * @return
 * Returns normalization matrix:
 * [2/w		0		-1]
 * [0		2/h		-1
 * [0		0		1]
 */
cv::matrixr normalize_image_points(std::vector<std::vector<cv::vec2r> > &patterns, unsigned w, unsigned h);

/*!
 * @brief Denormalize intrinsic matrix.
 *
 * @param A_p intrincis matrix calculated using normalized image points.
 * @param N Normalization matrix composed using normalize_image_points function.
 *
 * @return
 * De-normalized intrinsic matrix.
 */
cv::matrixr denormalize_intrinsics(const cv::matrixr &A_p, const cv::matrixr &N);

/*!
 * @brief Calculate reprojection error using given intrinsic and extrinsic matrices, and image and world points
 * used to calculate those.
 *
 * @param A intrinsic 3x3 matrix
 * @param K extrinsic [r1 r2 r3 t] 3x4 matrix.
 * @param model_pts world points of the pattern used for calibration.
 * @param image_pts image points of the pattern used for calibration.
 * @param image_pts_proj vector of points where reprojected points will be stored.
 * @param camera_pts camera projection points - K * model
 * 
 * @return 
 * square error of reprojection.
 */
real_t calc_reprojection_error(const cv::matrixr &A, const cv::matrixr &K,
                               const std::vector<cv::vec3r> &model_pts, const std::vector<cv::vec2r> &image_pts,
                               std::vector<cv::vec2r> &image_pts_proj, std::vector<cv::vec2r> &camera_pts);

/*!
 * @brief Compute intrinsic 3x3 matrix A from set of homography matrices calculated using calibration patterns.
 *
 * @param Hs vector of 3x3 homography matrices.
 *
 * @return
 * 3x3 intrinsic matrix A.
 */
cv::matrixr compute_intrisics(const std::vector<cv::matrixr> &Hs);

/*!
 * @brief Compute extrinsic 3x4 matrix using intrisic matrix and homogaphy matrix.
 * 
 * @param A intrinsic 3x3 matrix.
 * @param H homography which relates one of pattern image corner points to world points of the pattern.
 *
 * @return
 * 3x4 extrinsic matrix.
 */
cv::matrixr compute_extrinsics(const cv::matrixr &A, const cv::matrixr &H);

/*!
 * @brief Compute lens distortion parameters k1, and k2.
 *
 * @param image_pts vector of calibration pattern corners.
 * @param image_pts_nrm normalized copy of image_pts, used originally for calibration.
 * @param image_pts_proj reprojected image points, calculated using calc_reprojection_error function.
 * @param A intrinsic matrix of the camera.
 *
 * @return
 * 2d real vector with [k1, k2] as values.
 */
cv::vec2r compute_distortion(const std::vector<std::vector<cv::vec2r>> &image_pts, const std::vector<std::vector<cv::vec2r>> &image_pts_nrm,
                             const std::vector<std::vector<cv::vec2r>> &image_pts_proj, const cv::matrixr &A);


#endif /* end of include guard: CALIB_HPP_Q5H9DDXY */

