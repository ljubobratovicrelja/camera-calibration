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
	cv::matrixr translate, scale, transform;

	cv::vec2r centroid(0, 0);
	cv::matrixr S;

	for (auto feat : features) {
		centroid += feat;
	}
	centroid /= features.size();

	double avgDistance = 0;

	for (auto feat : features) {
		avgDistance += centroid.distance(feat);
	}

	avgDistance /= features.size();

	//scale.scale(sqrt(2) / avgDistance);
	//translate.translate(centroid * -1);

	transform = scale * translate;

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
	tgtS = DLT_pointSimilarityEstimation(tgt_pts);

	auto src_n = src_pts; // source normalized points
	auto tgt_n = tgt_pts; // target normalized points

	invTgtS = tgtS.clone();
	invert(invTgtS);

	DLT_normalize(src_n, srcS);
	DLT_normalize(tgt_n, tgtS);

	// 2. Pack matrix A;
	#pragma omp parallel
	{
		#pragma omp for
		for (unsigned i = 0; i < src_pts.size(); ++i) {
			// [-x -y -1 0 0 0 ux uy u]
			// [0 0 0 -x -y -1 vx vy v]

			A(i * 2 + 0, 0) = -1 * src_pts[i][0];
			A(i * 2 + 0, 1) = -1 * src_pts[i][1];
			A(i * 2 + 0, 2) = -1;
			A(i * 2 + 0, 6) = tgt_pts[i][0] * src_pts[i][0];
			A(i * 2 + 0, 7) = tgt_pts[i][0] * src_pts[i][1];
			A(i * 2 + 0, 8) = tgt_pts[i][0];

			A(i * 2 + 1, 3) = -1 * src_pts[i][0];
			A(i * 2 + 1, 4) = -1 * src_pts[i][1];
			A(i * 2 + 1, 5) = -1;
			A(i * 2 + 1, 6) = tgt_pts[i][1] * src_pts[i][0];
			A(i * 2 + 1, 7) = tgt_pts[i][1] * src_pts[i][1];
			A(i * 2 + 1, 8) = tgt_pts[i][1];
		}
	}

	// 3. solve nullspace of A for H;
	cv::null_solve(A, H);

	H.reshape(3, 3);

	// 4. denormalize the homography.
	H = invTgtS * H * srcS;
}


