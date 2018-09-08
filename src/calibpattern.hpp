#ifndef CALIBPATTERN_HPP_ORZQNP5Z
#define CALIBPATTERN_HPP_ORZQNP5Z

#include <vector>

#include <contour.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <draw.hpp>

std::vector<cv::vec2r> detect_pattern(const cv::matrixr& image,
                                      unsigned patternRows,
                                      unsigned patternCols,
                                      real_t angThresh = 15.,
                                      real_t magThresh = 0.15,
                                      unsigned nnCount = 10);

std::vector<cv::vec2r> sub_pixel_detect(const std::vector<cv::vec2r>& corners,
                                        const cv::matrixr& src,
                                        const cv::vec2i& win = {10,
                                                                10},
                                        real_t eps = 10e-6,
                                        unsigned maxIters = 100);

template<typename _Tp>
void draw_chessboard(cv::matrix<_Tp>& image, const std::vector<cv::vec2r>& pattern, const _Tp& color) {
    ASSERT(image && !pattern.empty());

    for (unsigned i = 1; i < pattern.size(); ++i) {
        cv::draw_line(image, pattern[i - 1], pattern[i], color);
    }
}

#endif /* end of include guard: CALIBPATTERN_HPP_ORZQNP5Z */
