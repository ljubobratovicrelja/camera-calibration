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
// Collection of functions for calibration pattern detection.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef CALIBPATTERN_HPP_ORZQNP5Z
#define CALIBPATTERN_HPP_ORZQNP5Z

#include <vector>

#include <contour.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <draw.hpp>

std::vector<cv::vec2r> detect_pattern(const cv::matrixr &image, unsigned patternRows, unsigned patternCols, 
		real_t angThresh = 15., real_t magThresh = 0.15, unsigned nnCount = 8);

std::vector<cv::vec2r> sub_pixel_detect(const std::vector<cv::vec2r> &corners, const cv::matrixr &src, 
		const cv::vec2i &win = {10, 10}, real_t eps = 10e-6, unsigned maxIters = 100);

template<typename _Tp>
void draw_chessboard(cv::matrix<_Tp> &image, const std::vector<cv::vec2r> &pattern, const _Tp &color) {

	ASSERT(image && !pattern.empty());

	for (unsigned i = 1; i < pattern.size(); ++i) {
		cv::draw_line(image, pattern[i-1], pattern[i], color);
	}
}


#endif /* end of include guard: CALIBPATTERN_HPP_ORZQNP5Z */


