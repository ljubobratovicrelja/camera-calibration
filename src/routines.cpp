#include "routines.h"
#include "improc.hpp"
#include "calibpattern.hpp"
#include "calibio.h"


void pattern_detection(const std::vector<cv::matrixr> &pattern_ims, const std::string &out_path, unsigned p_rows, unsigned p_cols, real_t ang_thresh, real_t mag_thresh, unsigned nn_count) {

	std::vector<std::vector<cv::vec2r> > patterns;

	unsigned im_w = 0, im_h = 0;

	for (auto image: pattern_ims) {

		im_w = image.cols();
		im_h = image.rows();

		cv::matrixr im_r = image, im_rb;
		cv::matrixr gauss_k = cv::gauss({3, 3}, 3);

		im_rb = cv::conv(im_r, gauss_k);

		auto pattern = detect_pattern(im_rb, p_rows, p_cols, ang_thresh, mag_thresh, nn_count);

		if (pattern.size() == p_rows*p_cols) {
			pattern = sub_pixel_detect(pattern, im_r, {5, 5});

			cv::matrix3r im_draw = im_r.clone();
			draw_chessboard(im_draw, pattern, cv::vec3r(255.f, 0.f, 0.f));

#ifndef CV_IGNORE_GUI
			std::cout << "Accept pattern? (Y/n)" << std::endl;

			cv::imshow("pattern", im_draw);
			auto r = cv::wait_key();

			if (r == 'n') {
				std::cout << "Pattern rejected." << std::endl;
				continue;
			} else {
				std::cout << "Pattern accepted." << std::endl;
				patterns.push_back(pattern);
			}
#else
			std::cout << "Pattern found." << std::endl;
			patterns.push_back(pattern);
#endif
		} else {
			std::cout << "Pattern not found" << std::endl;
			continue;
		}
	}

	if (write_pattern_results(patterns, im_w, im_h, out_path.c_str())) {
		std::cout << "Writing results to " << out_path << " successfull!" << std::endl;
	} else {
		std::cout << "Writing results to " << out_path << " failed!" << std::endl;
	}
}
