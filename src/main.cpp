#include <io.hpp>
#include <gui.hpp>
#include <image.hpp>
#include <improc.hpp>
#include <region.hpp>
#include <contour.hpp>
#include <draw.hpp>

#include <iostream>

#include "calibpattern.hpp"

void pattern_detection() {

	auto image = cv::imread("/home/relja/calibration/ground_truth/2.png", cv::REAL, 1);

	unsigned p_rows = 6;
	unsigned p_cols = 9;

	cv::matrixr im_r = image, im_rb;
	cv::matrixr gauss_k = cv::gauss({3, 3});

	im_rb = cv::conv(im_r, gauss_k);

	auto pattern = detect_pattern(im_rb, p_rows, p_cols, 18., 0.15);

	if (pattern.size() == p_rows*p_cols) {

		pattern = sub_pixel_detect(pattern, im_r, {5, 5});

		cv::matrix3r im_draw = im_r.clone();
		draw_chessboard(im_draw, pattern, {255., 0., 0.});

		cv::imshow("pattern", im_draw);
		cv::wait_key();
	} else {
		std::cout << "Pattern not found" << std::endl;
		return;
	}
}

int main() {

	pattern_detection();
	
	return 0;
}
