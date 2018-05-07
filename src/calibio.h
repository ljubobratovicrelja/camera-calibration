#pragma once
#include <vector>
#include <image.hpp>


bool
write_pattern_results
(
	const std::vector<std::vector<cv::vec2r> > &patterns,
	unsigned im_w,
	unsigned im_h,
	const char *path
);

std::vector<std::vector<cv::vec2r> >
read_pattern_results
(
	const char *path,
	unsigned &im_w,
	unsigned &im_h
);

bool
read_zhang_data
(
	const std::string &folderpath,
	std::vector<std::vector<cv::vec2r> > &image_points,
	std::vector<cv::vec3r > &model_points,
	unsigned &im_w,
	unsigned &im_h
);

std::vector<std::string>
read_image_collection(const std::string &file);

bool
read_custom_data
(
	const std::string &filepath,
	std::vector<std::vector<cv::vec2r> > &image_points,
	std::vector<cv::vec3r > &model_points,
	unsigned &im_w,
	unsigned &im_h,
	double model_size
);

void
write_results
(
	const cv::matrixr &A,
	const cv::vectorr &k,
	const std::vector<cv::matrixr> Ks,
	const std::string &path = "."
);
