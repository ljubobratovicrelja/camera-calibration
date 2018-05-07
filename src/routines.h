#pragma once
#include <vector>
#include <matrix.hpp>


/**
    Pattern detection routine.

    @param pattern_ims Vector of pattern projections (images).
    @param out_path File path where pattern detection coordinates will be stored.
    @param p_rows How many rows does the pattern to be looked contain?
    @param p_cols How many columns does the pattern to be looked contain?
    @param ang_thresh Angular threshold, parameter defining straight lines in detection algorithm.
    @param mag_thresh Distance between two neighbouring points in the pattern.
    @param nn_count Number of neighbouring points to be considered while searching the pattern.
*/
void
pattern_detection
(
    const std::vector<cv::matrixr> &pattern_ims,
    const std::string &out_path,
    unsigned p_rows,
    unsigned p_cols,
    real_t ang_thresh,
    real_t mag_thresh,
    unsigned nn_count
);
