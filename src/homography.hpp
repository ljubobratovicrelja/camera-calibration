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


/*
 * Solve homography using non-normalized 8-point algorithm.
 */
void homography_solve(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, cv::matrixr &H);

/*
 * Normalized Direct linear transformation algorithm for homography estimation.
 */
void homography_dlt(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec3r> &tgt_pts, cv::matrixr &H);

/*
 * Solve homography using least squares method.
 */
void homography_least_squares(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec3r> &tgt_pts, cv::matrixr &H);

/*!
 * @brief Optimization routine collection using Levenberg-Marquadt algorithm.
 */
struct homography_optimization {

	static std::vector<cv::vec2r> source_pts; //!< Source points with which the initial homography was estimated.
	static std::vector<cv::vec3r> target_pts; //!< Target points with which the initial homography was estimated.

	//! Reprojection error function.
	static void reprojection_fcn(int *m, int *n, double* x, double* fvec,int *iflag);

	//! Evaluate optmization for given data set with given function.
	static int evaluate(cv::matrixr &H, cv::optimization_fcn fcn, double tol = 10e-32);
};


#endif /* end of include guard: HOMOGRAPHY_HPP_ERDRJZXL */
