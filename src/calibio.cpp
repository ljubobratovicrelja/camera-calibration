#include "calibio.h"

#include <fstream>
#include <iomanip>

#include "utils.h"
#include "calib.hpp"


bool write_pattern_results(const std::vector<std::vector<cv::vec2r> > &patterns, unsigned im_w, unsigned im_h, const char *path) {

	std::ofstream stream(path);

	if (!stream.is_open()) {
		return false;
	}

	stream << im_w << " " << im_h << std::endl;

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

std::vector<std::vector<cv::vec2r> > read_pattern_results(const char *path, unsigned &im_w, unsigned &im_h) {
	std::vector<std::vector<cv::vec2r> > patterns;

	std::ifstream stream(path);

	if(!stream.is_open()) {
		return patterns;
	}

	std::string line;
	if(std::getline(stream, line)) {
		auto l = split_string(line, " ");
		if (l.size() == 2) {
			im_w = std::atoi(l[0].c_str());
			im_h = std::atoi(l[1].c_str());
		}
	} else {
		im_w = im_h = 0;
		return patterns;
	}

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

bool read_zhang_data(const std::string &folderpath, std::vector<std::vector<cv::vec2r> > &image_points,
                     std::vector<cv::vec3r > &model_points, unsigned &im_w, unsigned &im_h) {
	int i,n=0;
	FILE* fpm  = fopen((folderpath + "/model.txt").c_str(),"rt");
	FILE* fpi1 = fopen((folderpath + "/data1.txt").c_str(),"rt");
	FILE* fpi2 = fopen((folderpath + "/data2.txt").c_str(),"rt");
	FILE* fpi3 = fopen((folderpath + "/data3.txt").c_str(),"rt");
	FILE* fpi4 = fopen((folderpath + "/data4.txt").c_str(),"rt");
	FILE* fpi5 = fopen((folderpath + "/data5.txt").c_str(),"rt");

	if (fpi1==NULL ||fpi2==NULL ||fpi3==NULL ||fpi4==NULL ||fpi5==NULL || fpm==NULL) {
		printf("Arq error\n");
		return 1;
	}

	for (n=0; !feof(fpm); n++ ) {
		double x, y;
		fscanf(fpm,"%lf %lf ",&x,&y);
		model_points.push_back(cv::vec3r(x, y, 1.));
	}

	fclose(fpm);

	image_points.resize(5);
	for (i=0; i<n; i++ ) {
		double x, y;
		fscanf(fpi1,"%lf %lf ",&x,&y);
		image_points[0].push_back(cv::vec2r(x, y));
		fscanf(fpi2,"%lf %lf ",&x,&y);
		image_points[1].push_back(cv::vec2r(x, y));
		fscanf(fpi3,"%lf %lf ",&x,&y);
		image_points[2].push_back(cv::vec2r(x, y));
		fscanf(fpi4,"%lf %lf ",&x,&y);
		image_points[3].push_back(cv::vec2r(x, y));
		fscanf(fpi5,"%lf %lf ",&x,&y);
		image_points[4].push_back(cv::vec2r(x, y));
	}

	fclose(fpi1);
	fclose(fpi2);
	fclose(fpi3);
	fclose(fpi4);
	fclose(fpi5);

	im_w = 640;
	im_h = 480;

	return true;
}

std::vector<std::string> read_image_collection(const std::string &file) {

	std::vector<std::string> im_files;
	std::ifstream stream(file.c_str());

	if (stream.is_open())  {
		std::string line;
		while (std::getline(stream, line)) {
			im_files.push_back(line);
		}
		stream.close();
	}

	return im_files;
}

bool read_custom_data(const std::string &filepath, std::vector<std::vector<cv::vec2r> > &image_points,
                      std::vector<cv::vec3r > &model_points, unsigned &im_w, unsigned &im_h, double model_size) {
	image_points = read_pattern_results(filepath.c_str(), im_w, im_h);
	model_points = calculate_object_points(6, 9, model_size);

	return true;
}

void write_results(const cv::matrixr &A, const cv::vectorr &k, const std::vector<cv::matrixr> Ks, const std::string &path) {

	std::ofstream a_stream(path + "/a.out");
	a_stream << std::setprecision(std::numeric_limits<real_t>::digits10+1);

	auto a = A.data();
	unsigned i;

	a_stream << a[0];
	for (i = 1; i < 9; ++i) {
		a_stream << "," << a[i];
	}

	a_stream << std::endl;

	if (k) {
		a_stream << k[0];
		for (i = 1; i < k.length(); ++i) {
			a_stream << "," << k[i];
		}
	}

	a_stream << std::endl;

	for (auto K : Ks) {
		auto k = K.data();
		a_stream << k[0];
		for (i = 1; i < 12; ++i) {
			a_stream << "," << k[i];
		}

		a_stream << std::endl;
	}
}
