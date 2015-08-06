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
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#include "homography.hpp"

#include <linalg.hpp>
#include <math.hpp>


cv::matrixr DLT_pointSimilarityEstimation(const std::vector<cv::vec2r> &features) {
	cv::matrixr transform = cv::matrixr::eye(3);

	cv::vec2r centroid(0, 0);
	cv::matrixr S;

	for (auto feat : features) {
		centroid += feat;
	}
	centroid /= features.size();

	double sum_dist = 0;

	for (auto feat : features) {
		sum_dist+= centroid.distance(feat);
	}
	centroid *= -1;

	real_t scale_v = std::sqrt(2.) / (sum_dist / features.size());

	transform(0, 0) = scale_v;
	transform(1, 1) = scale_v;
	transform(0, 2) = centroid[0];
	transform(1, 2) = centroid[1];

	return transform;
}

void DLT_normalize(std::vector<cv::vec2r> &features, const cv::matrixr &S) {
	ASSERT(S && S.rows() == 3 && S.cols() == 3);
	cv::matrixr x(3, 1), xp(3, 1);
	for (unsigned i = 0; i < features.size(); ++i) {
		x(0, 0) = features[i][0];
		x(1, 0) = features[i][1];
		x(2, 0) = 1;
		cross(S, x, xp);
		features[i][0] = xp(0, 0);
		features[i][1] = xp(1, 0);
	}
}

void DLT(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec2r> &tgt_pts, cv::matrixr &H) {
	ASSERT(src_pts.size() >= 4 && src_pts.size() == tgt_pts.size());

	// 0. Prepare data;
	cv::matrixr srcS, tgtS, invTgtS;
	cv::matrixr A = cv::matrixr::zeros(2 * src_pts.size(), 9);

	// 1. Perform normalization;
	srcS = DLT_pointSimilarityEstimation(src_pts);
	std::cout << srcS << std::endl;
	tgtS = DLT_pointSimilarityEstimation(tgt_pts);

	auto src_n = src_pts; // source normalized points
	auto tgt_n = tgt_pts; // target normalized points

	invTgtS = tgtS.clone();
	invert(invTgtS);

	DLT_normalize(src_n, srcS);
	DLT_normalize(tgt_n, tgtS);

	// 2. Pack matrix A;
	for (unsigned i = 0; i < src_pts.size(); ++i) {
		// [-x -y -1 0 0 0 ux uy u]
		// [0 0 0 -x -y -1 vx vy v]

		A(i * 2 + 0, 0) = -1 * src_n[i][0];
		A(i * 2 + 0, 1) = -1 * src_n[i][1];
		A(i * 2 + 0, 2) = -1;
		A(i * 2 + 0, 6) = tgt_n[i][0] * src_n[i][0];
		A(i * 2 + 0, 7) = tgt_n[i][0] * src_n[i][1];
		A(i * 2 + 0, 8) = tgt_n[i][0];

		A(i * 2 + 1, 3) = -1 * src_n[i][0];
		A(i * 2 + 1, 4) = -1 * src_n[i][1];
		A(i * 2 + 1, 5) = -1;
		A(i * 2 + 1, 6) = tgt_n[i][1] * src_n[i][0];
		A(i * 2 + 1, 7) = tgt_n[i][1] * src_n[i][1];
		A(i * 2 + 1, 8) = tgt_n[i][1];
	}

	// 3. solve nullspace of A for H;
	cv::null_solve(A, H);

	H.reshape(3, 3);

	// 4. denormalize the homography.
	H = invTgtS * H * srcS;
}

/*
 * Pack homography matrices A and B by the form used for least squares solving.
 */
void packHomographyAB(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec2r> &tgt_pts, cv::matrixr &A, cv::matrixr &B) {

	ASSERT(src_pts.size() && src_pts.size() == tgt_pts.size());

	// construct matrices
	A = cv::matrixr::zeros(src_pts.size() * 2, 8);
	B.create(src_pts.size() * 2, 1);

	// populate matrices with data.
	for (unsigned i = 0; i < src_pts.size(); i++) {

		auto &src = src_pts[i];
		auto &tgt = tgt_pts[i];

		B(i * 2, 0) = tgt[0];
		B(i * 2 + 1, 0) = tgt[1];

		A(i * 2, 0) = src[0];
		A(i * 2, 1) = src[1];
		A(i * 2, 2) = 1;
		A(i * 2 + 1, 3) = src[0];
		A(i * 2 + 1, 4) = src[1];
		A(i * 2 + 1, 5) = 1;

		A(i * 2, 6) = -1 * src[0] * tgt[0];
		A(i * 2, 7) = -1 * src[1] * tgt[0];
		A(i * 2 + 1, 6) = -1 * src[0] * tgt[1];
		A(i * 2 + 1, 7) = -1 * src[1] * tgt[1];
	}
}

/*
 * Solve homography using least squares method.
 */
void solveLeastSquaresHomography(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec2r> &tgt_pts, cv::matrixr &H) {

	cv::matrixr A, B;
	packHomographyAB(src_pts, tgt_pts, A, B);
	cv::matrixr _H(8, 1);

	cv::matrixr At = A.transposed();
	lu_solve(At * A, At * B, _H);

	if (!_H) {
		throw std::runtime_error("Internal error!~ failure occurred calculating homography.\n");
	}

	H.create(1, 9);
	std::copy(_H.begin(), _H.end(), H.begin());
	H.reshape(3, 3);
	H(2, 2) = 1;

}

