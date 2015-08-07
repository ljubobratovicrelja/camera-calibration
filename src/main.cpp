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


#include <io.hpp>
#include <gui.hpp>
#include <image.hpp>
#include <improc.hpp>
#include <region.hpp>
#include <contour.hpp>
#include <draw.hpp>
#include <linalg.hpp>
#include <optimization.hpp>
#include <matfunc.hpp>

#include <iostream>
#include <fstream>

#include <minpack.h>

#include "calibpattern.hpp"
#include "homography.hpp"

std::vector<std::string> split_string(std::string s, const std::string &delimiter = " ") {
	std::vector<std::string> tokens;

	size_t pos = 0;
	std::string token;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		tokens.push_back(s.substr(0, pos));
		s.erase(0, pos + delimiter.length());
	}
	tokens.push_back(s);

	return tokens;
}

bool write_pattern_results(const std::vector<std::vector<cv::vec2r> > &patterns, const char *path) {

	std::ofstream stream(path);

	if (!stream.is_open()) {
		return false;
	}

	for (unsigned i = 0; i < patterns.size(); ++i) {
		stream << patterns[i][0];
		for (unsigned j = 1; j < patterns[i].size(); ++j) {
			stream << "," << patterns[i][j];
		}
		stream << std::endl;
	}

	stream.close();

	return true;
}

std::vector<std::vector<cv::vec2r> > read_pattern_results(const char *path) {
	std::vector<std::vector<cv::vec2r> > patterns;

	std::ifstream stream(path);

	if(!stream.is_open()) {
		return patterns;
	}

	std::string line;
	while(std::getline(stream, line)) {
		patterns.push_back(std::vector<cv::vec2r>());
		auto &p = patterns.back();
		for (auto &v : split_string(line, ",")) {
			auto vec = split_string(v, " ");
			if (vec.size() == 2) {
				p.push_back(cv::vec2r(static_cast<real_t>(std::atof(vec[0].c_str())), static_cast<real_t>(std::atof(vec[1].c_str()))));
			}
		}
	}

	stream.close();

	return patterns;
}

void pattern_detection() {

	std::string calib_patterns[] = {"1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png"};

	std::vector<std::vector<cv::vec2r> > patterns;

	for (auto im_path : calib_patterns) {

		auto image = cv::imread("/home/relja/calibration/ground_truth/" + im_path, cv::REAL, 1);

		unsigned p_rows = 6;
		unsigned p_cols = 9;

		cv::matrixr im_r = image, im_rb;
		cv::matrixr gauss_k = cv::gauss({5, 5});

		im_rb = cv::conv(im_r, gauss_k);

		auto pattern = detect_pattern(im_rb, p_rows, p_cols, 18., 0.15, 12);

		if (pattern.size() == p_rows*p_cols) {

			pattern = sub_pixel_detect(pattern, im_r, {5, 5});

			cv::matrix3r im_draw = im_r.clone();
			draw_chessboard(im_draw, pattern, {255., 0., 0.});

			cv::imshow("pattern", im_draw);
			auto r = cv::wait_key();

			if (r == 'n') {
				std::cout << "Pattern " << im_path << " rejected." << std::endl;
				continue;
			} else if (r == 'q') {
				std::cout << "Pattern detection exit" << std::endl;
				break;
			} else {
				std::cout << "Pattern " << im_path << " accepted." << std::endl;
				patterns.push_back(pattern);
			}
		} else {
			std::cout << "Pattern " << im_path << " not found" << std::endl;
			continue;
		}
	}

	std::string out_path;
	std::cout << "Where to write results?" << std::endl;
	std::cin >> out_path;

	if (write_pattern_results(patterns, out_path.c_str())) {
		std::cout << "Writing results to " << out_path << " successfull!" << std::endl;
	} else {
		std::cout << "Writing results to " << out_path << " failed!" << std::endl;
	}
}

cv::matrixr calculate_homography() {
	cv::matrixr H;
	return H;
}

std::vector<cv::vec2r> calculate_object_points(unsigned rows, unsigned cols, real_t square_size) {
	std::vector<cv::vec2r> obj_pts;
	obj_pts.reserve(rows*cols);

	for(unsigned i = 0; i < rows; ++i) {
		for(unsigned j = 0; j < cols; ++j) {
			obj_pts.push_back({static_cast<real_t>(j*square_size), static_cast<real_t>(i*square_size)});
		}
	}

	return obj_pts;
}

real_t calc_reprojection_error(const cv::matrixr &H, const std::vector<cv::vec2r> &source_pts, const std::vector<cv::vec2r> &target_pts) {

	ASSERT(source_pts.size() == target_pts.size() && H && H.rows() == 3 && H.cols() == 3);

	unsigned ptn_count = source_pts.size();
	real_t err = 0.0;

	// calculate m_projected
	cv::matrixr ptn(3, 1), p_ptn(3, 1), res_ptn(3, 1);

	for (unsigned i = 0; i < ptn_count; ++i) {
		ptn(0, 0) = source_pts[i][0];
		ptn(1, 0) = source_pts[i][1];
		ptn(2, 0) = 1.;

		p_ptn(0, 0) = target_pts[i][0];
		p_ptn(1, 0) = target_pts[i][1];
		p_ptn(2, 0) = 1.;

		cv::cross( H, ptn, res_ptn);

		err += pow(cv::distance(res_ptn, p_ptn, cv::Norm::L2), 2);
	}

	return sqrt(err);
}

cv::vectorr get_vij(const cv::matrixr &h, unsigned i, unsigned j) {
	return cv::vectorr {
		h(0, i)*h(0, j), h(0, i)*h(1, j) + h(1, i)*h(0, j), h(1, i)*h(1, j),
		h(2, i)*h(0, j) + h(0, i)*h(2, j), h(2, i)*h(1, j) + h(1, i)*h(2, j), h(2, i)*h(2, j)
	};
}

cv::matrixr pack_v(const std::vector<cv::matrixr> &Hs) {
	cv::matrixr v(2*Hs.size(), 6);

	for (unsigned i = 0; i < Hs.size(); ++i) {

		auto h_r_1 = v.row(i*2);
		auto h_r_2 = v.row(i*2 + 1);

		auto v12 = get_vij(Hs[i], 0, 1);
		auto v11 = get_vij(Hs[i], 0, 0);
		auto v22 = get_vij(Hs[i], 1, 1);

		std::copy(h_r_1.begin(), h_r_1.end(), v12.begin());
		std::copy(h_r_2.begin(), h_r_2.end(), (v11 - v22).begin());
	}

	return v;
}

cv::vectorr solve_b(const cv::matrixr &V) {

	cv::matrixr VtV;
	cv::cross(V.transposed(), V, VtV);

	cv::vectorr we, wi;
	cv::matrixr vl, vr;

	cv::geev(VtV, we, wi, vl, vr);
	std::vector<cv::vectorr> eig_vecs;

	cv::decompose_eigenvector_matrix(vr, eig_vecs, true);

	return eig_vecs[std::distance(we.begin(), we.min())];
}

int main() {

	//pattern_detection();
	//
	auto patterns = read_pattern_results("/home/relja/git/camera_calibration/pattern.txt");
	auto model_points = calculate_object_points(6, 9, 1.0);

	auto patterns_count = patterns.size();

	std::vector<cv::matrixr> Hs(patterns_count);

	for (unsigned i = 0; i < patterns_count; ++i) {
		ASSERT(patterns[i].size() == model_points.size());

		auto &H = Hs[i];

		homography_least_squares(patterns[i], model_points, H);

		std::cout << "Initial homography:" << std::endl;
		std::cout << H << std::endl << std::endl;
		std::cout << "Initial homography calculation error: " << calc_reprojection_error(H, patterns[i], model_points) << std::endl;

		homography_optimization::source_pts = patterns[i];
		homography_optimization::target_pts = model_points;
		homography_optimization::evaluate(H, homography_optimization::reprojection_fcn);

		std::cout << "Homography after geometric error minimization:" << std::endl;
		std::cout << H << std::endl;
		std::cout << "After geometric optimization, homography calculation error: " << calc_reprojection_error(H, patterns[i], model_points) << std::endl;
	}

	auto V = pack_v(Hs);
	auto b = solve_b(V);

	std::cout << "V matrix:" << std::endl;
	std::cout << V << std::endl;
	std::cout << "b vector:" << std::endl;
	std::cout << b << std::endl;
	return 0;
}


