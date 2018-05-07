#pragma once

#include <vector>
#include <string>

#include <image.hpp>

/// Split a string by a given delimiter.
std::vector<std::string>
split_string
(
    std::string string,
    const std::string &delimiter = " "
);

cv::image_array
undistort_image
(
    const cv::image_array &image,
    const cv::matrixr &A,
    const cv::vectorr &k
);

/// String to double with error proofing.
void
string_to_double
(
    double &val,
    int argc,
    char **argv,
    int i,
    const std::string &val_name
);

/// String to int with error proofing.
void
string_to_int
(
    int &val,
    int argc,
    char **argv,
    int i,
    const std::string &val_name
);
