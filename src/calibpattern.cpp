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


#include "calibpattern.hpp"

#include <algorithm>
#include <memory>

#include <kdtree.hpp>
#include <improc.hpp>

// debug
#include <gui.hpp>
#include <draw.hpp>

// forward declaration ////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<unsigned> > find_nns(std::vector<cv::vec2r> &features, unsigned nncount);

std::vector<cv::contouri> detect_lines(std::vector<cv::vec2r> &features, std::vector<std::vector<unsigned> > &nns, real_t angleThresh, real_t magThresh, unsigned rows, unsigned cols);

cv::contouri detect_line(unsigned startIdx, unsigned queryIdx, std::vector<cv::vec2r> &features, std::vector<
                        std::vector<unsigned> >&nns, real_t angleThreshold, real_t magThreshold, unsigned rows, unsigned cols);


std::vector<cv::vec2r> detect_chessboard(const std::vector<cv::contouri> &contours, unsigned rows, unsigned cols);

// Implementation /////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename _Tp>
bool is_aproximation(const _Tp &query, const _Tp &source, const _Tp &error) {
	return (fabs(source - query) < error);
}

std::vector<std::vector<unsigned> > find_nns(std::vector<cv::vec2r> &features, unsigned nncount) {

	std::vector<std::vector<unsigned> > nns(features.size());

	cv::kd_tree2i kd;
	std::vector<cv::vec2i> data;

	for (auto & f : features) {
		data.push_back(f);
	}

	kd.set_data(data);
	kd.build();

	for (unsigned i = 0; i < features.size(); ++i) {
		auto &f = features[i];

		kd.knn_index(f, nncount, nns[i]);

		auto pos = std::find(nns[i].begin(), nns[i].end(), i);
		if (pos != nns[i].end()) {
			nns[i].erase(pos);
		}
        std::sort(nns[i].begin(), nns[i].end());
        nns[i].erase(std::unique(nns[i].begin(), nns[i].end()), nns[i].end());
	}

	return nns;
}

std::vector<cv::contouri> detect_lines(std::vector<cv::vec2r> &features, std::vector<std::vector<unsigned> > &nns, real_t angleThresh, real_t magThresh, unsigned rows, unsigned cols) {

	std::vector<cv::contouri> lines;

	for (unsigned i = 0; i < features.size(); i++) {
		for (unsigned j = 0; j < nns[i].size(); j++) {
			cv::contouri c = detect_line(i, nns[i][j], features, nns, angleThresh, magThresh, rows, cols);
			if (!c.empty()) {
				lines.push_back(c);
			}
		}
	}

	return lines;
}

cv::contouri detect_line(unsigned startIdx, unsigned queryIdx, std::vector<cv::vec2r> &features, std::vector<
                        std::vector<unsigned> >&nns, real_t angleThreshold, real_t magThreshold, unsigned rows, unsigned cols) {

	cv::contouri line;
	line.add_points( { features[startIdx], features[queryIdx] });

	while (true) {

		auto edgeVec = features[queryIdx] - features[startIdx];

		int bestId = -1;
		real_t bestAngle = 90.0f;
		real_t bestMagnitureError = 1.0f;

		for (auto nn : nns[queryIdx]) {
			
			if (nn == queryIdx or nn == startIdx) {
				continue;
			}

			auto queryVec = features[nn] - features[queryIdx];

			real_t ang = RAD_TO_DEG(edgeVec.angle(queryVec));
			real_t mag = 1.0 - ((edgeVec.norm() < queryVec.norm()) ? (edgeVec.norm() / queryVec.norm()) : (queryVec.norm() / edgeVec.norm()));

			if (ang < angleThreshold && mag < magThreshold && ang < bestAngle && mag < bestMagnitureError) {
				bestId = nn;
				bestAngle = ang;
				bestMagnitureError = mag;
			}
		}

		if (bestId != -1) {
			startIdx = queryIdx;
			queryIdx = bestId;
			line.add_point(features[bestId]);
		} else {
			break;
		}
	}

	unsigned line_size = line.point_length();

	if (line_size == rows or line_size == cols) {
		return line;
	} else {
		return cv::contouri();
	}
}

std::vector<cv::vec2r> sub_pixel_detect(const std::vector<cv::vec2r> &corners, const cv::matrixr &src, const cv::vec2i &win, real_t eps, unsigned maxIters) {

	std::vector<cv::vec2r> outCorners(corners.size());

	const int MAX_ITERS = 100;
	const cv::matrixr drv_x = {{-1., 0, 1.}};
	const cv::matrixr drv_y = drv_x.transposed();
	cv::matrixr gx;
	cv::matrixr gy;
	cv::matrixr src_d = src;
	cv::matrixr scr_kernel;
	real_t coeff;
	int i, j, k, pt_i;
	int win_w = win[1] * 2 + 1, win_h = win[0] * 2 + 1;

	if (eps < 0.)
		eps = 0.;
	eps *= eps; /* use square of error in comparsion operations. */

	unsigned max_iters = std::max((int) maxIters, 1);
	max_iters = std::min((int) max_iters, MAX_ITERS);

	coeff = 1. / (win[0] * win[1]);

	cv::vector<real_t> maskX(win_w), maskY(win_h);
	/* calculate mask */
	for (i = -win[1], k = 0; i <= win[1]; i++, k++) {
		maskX[k] = (real_t) exp(-i * i * coeff);
	}
	if (win[0] == win[1]) {
		maskY = maskX;
	} else {
		for (i = -win[0], k = 0; i <= win[0]; i++, k++) {
			maskY[k] = (real_t) exp(-i * i * coeff);
		}
	}

	cv::matrixr mask(win_h, win_w);

	for (i = 0; i < win_h; i++) {
		for (j = 0; j < win_w; j++) {
			mask(i, j) = maskX[j] * maskY[i];
		}
	}

	/* do optimization loop for all the points */
	for (pt_i = 0; pt_i < corners.size(); pt_i++) {
		cv::vec2r cT = (cv::vec2r) corners[pt_i], cI = cT;

		int iter = 0;
		real_t err;

		real_t a, b, c, bb1, bb2;

		do {
			cv::vec2r cI2;
			scr_kernel.create(cI[1] - win_h / 2, cI[0] - win_w / 2, win_h, win_w, src_d);

			/* calc derivatives */
			gx = cv::conv(scr_kernel, drv_x);
			gy = cv::conv(scr_kernel, drv_y);

			a = b = c = bb1 = bb2 = 0;

			for (i = 0; i < win_w; i++) {
				real_t py = i - win[0];

				for (j = 0; j < win_h; j++) {
					real_t m = mask(i, j);
					real_t tgx = gx(i, j);
					real_t tgy = gy(i, j);
					real_t gxx = tgx * tgx * m;
					real_t gxy = tgx * tgy * m;
					real_t gyy = tgy * tgy * m;
					real_t px = j - win[1];

					a += gxx;
					b += gxy;
					c += gyy;

					bb1 += gxx * px + gxy * py;
					bb2 += gxy * px + gyy * py;
				}
			}

			real_t det = a * c - b * b;
			if (fabs(det) > std::numeric_limits<real_t>::epsilon() * std::numeric_limits<real_t>::epsilon()) {
				real_t scale = 1.0 / det;
				cI2[0] = cI[0] + c * scale * bb1 - b * scale * bb2;
				cI2[1] = cI[1] - b * scale * bb1 + a * scale * bb2;
			} else {
				cI2 = cI;
			}

			err = (cI2[0] - cI[0]) * (cI2[0] - cI[0]) + (cI2[1] - cI[1]) * (cI2[1] - cI[1]);
			cI = cI2;
		} while (++iter < max_iters && err > eps);
		if (fabs(cI[0] - cT[0]) > win[1] || fabs(cI[1] - cT[1]) > win[0]) {
			cI = cT;
		}
		outCorners[pt_i] = cI; /* store result */
	}

	return outCorners;
}

std::vector<cv::vec2r> detect_chessboard(const std::vector<cv::contouri> &contours, unsigned rows, unsigned cols) {
	std::vector<cv::vec2r> chessboard_corners;

	std::vector<cv::contouri> hip_chess;
	for (auto c : contours) {
		if (c.point_length() == rows) {
			hip_chess.clear();

			cv::contouri sorted_c(c);
			sorted_c.sort_by_axis(1);  // sort by y

			for (auto p : sorted_c) {
				for (unsigned i = 0; i < contours.size(); ++i) {
					if (contours[i].point_length() != cols) {
						continue;
					}
					cv::vec2r c_q_vec = contours[i].get_contour_vector();
					if ((p == contours[i][0] or p == contours[i][contours[i].point_length()-1])) {
						if (hip_chess.empty()) {
							hip_chess.push_back(contours[i]);
						} else {
							real_t mean_mag = 0.0;
							for (auto hip : hip_chess) {
								mean_mag += hip.get_contour_vector().norm();
							}
							mean_mag /= (real_t) hip_chess.size();
							if (is_aproximation<real_t>(c_q_vec.norm(), mean_mag, mean_mag / 5.0)) {
								hip_chess.push_back(contours[i]);
							}
						}
						break;
					}
				}
			}
			if (hip_chess.size() == rows)
				break;
		}
	}

	if (hip_chess.size() == rows) {
		for (unsigned i = 0; i < rows; i++) {
			std::vector<cv::vec2i> row_corners;
			for (unsigned j = 0; j < cols; j++) {
				row_corners.push_back(hip_chess[i][j]);
			}
			std::sort(row_corners.begin(), row_corners.end(), cv::internal::idx_cmp(0));
			for (auto rc : row_corners) {
				chessboard_corners.push_back(rc);
			}
		}
	}

	return chessboard_corners;
}

std::vector<cv::vec2r> detect_pattern(const cv::matrixr &image, unsigned patternRows, unsigned patternCols, real_t angThresh, real_t magThresh, unsigned nnCount) {

	std::vector<cv::vec2r> pattern;
	std::vector<std::vector<unsigned> > nns;
	std::vector<cv::contouri> ctns;
	std::vector<cv::vec2r> features;

	auto h_c = cv::good_features(image, 7);
	cv::filter_non_maximum(h_c, 20);
	features = cv::extract_features(h_c, patternCols*patternRows*1.5);

	nns = find_nns(features, nnCount);
	ctns = detect_lines(features, nns, angThresh, magThresh, patternRows, patternCols);
	pattern = detect_chessboard(ctns, patternRows, patternCols);

	return pattern;
}


