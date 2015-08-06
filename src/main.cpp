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

#include <iostream>
#include <fstream>

#include <minpack.h>

#include "calibpattern.hpp"
#include "homography.hpp"

bool write_pattern_results(const std::vector<std::vector<cv::vec2r> > &patterns, const std::string &path) {

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

	if (write_pattern_results(patterns, out_path)) {
		std::cout << "Writing results to " << out_path << " successfull!" << std::endl;
	} else {
		std::cout << "Writing results to " << out_path << " failed!" << std::endl;
	}
}

cv::matrixr calculate_homography() {
	cv::matrixr H;
	return H;
}

std::vector<cv::vec3r> calculate_object_points(unsigned rows, unsigned cols, real_t square_size) {
	std::vector<cv::vec3r> obj_pts;
	obj_pts.reserve(rows*cols);

	for(unsigned i = 0; i < rows; ++i) {
		for(unsigned j = 0; j < cols; ++j) {
			obj_pts.push_back({static_cast<real_t>(j*square_size), static_cast<real_t>(i*square_size), static_cast<real_t>(0.)});
		}
	}

	return obj_pts;
}	

real_t calc_homography_error(const cv::matrixr &H, real_t x_offset, real_t y_offset) {

	real_t err = 0.0;

	err += fabs(1.0 - H(0, 0));
	err += fabs(1.0 - H(1, 1));
	err += fabs(1.0 - H(2, 2));

	err += fabs(x_offset - H(0, 2));
	err += fabs(y_offset - H(1, 2));

	err += fabs(0.0 - H(0, 1));
	err += fabs(0.0 - H(1, 0));

	err += fabs(0.0 - H(2, 0));
	err += fabs(0.0 - H(2, 1));

	return err;
}

int main() {

	//pattern_detection();
	std::vector<cv::vec2r> start = {{30., 10}, {12, 30}, {93, 12}, {12, 32}, {7, 5}, {12, 98}, {123, 543}, {321, 324}, {43, 65}, {123, 43}};
	std::vector<cv::vec2r> end;

	real_t x_offset = 10.0;
	real_t y_offset = 0.;
	
	end = start;

	for (auto &v : end) {
		v += {x_offset, y_offset};
	}

	for (auto i : start) {
		std::cout << i << " ";
	}
	std::cout << std::endl;
	for (auto i : end) {
		std::cout << i << " ";
	}
	std::cout << std::endl;

	cv::matrixr H;

	DLT(start, end, H);

	std::cout << "Initial homography:" << std::endl;
	std::cout << H << std::endl << std::endl;
	std::cout << "Initial homography calculation error: " << calc_homography_error(H, x_offset, y_offset) << std::endl;

	homography_optimization::source_pts = start;
	homography_optimization::target_pts = end;
	homography_optimization::evaluate(H, homography_optimization::reprojection_fcn);

	std::cout << "Homography after geometric error minimization:" << std::endl;
	std::cout << H << std::endl << std::endl;
	std::cout << "After geometric optimization, homography calculation error: " << calc_homography_error(H, x_offset, y_offset) << std::endl;
	
	return 0;
}
