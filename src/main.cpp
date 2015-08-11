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

#include <iostream>
#include <fstream>

#include "calibpattern.hpp"
#include "homography.hpp"
#include "calib.hpp"


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
		auto &p = patterns.back();
		for (auto &v : split_string(line, ",")) {
			auto vec = split_string(v, " ");
			if (vec.size() == 2) {
				p.push_back(cv::vec2r(static_cast<real_t>(std::atof(vec[0].c_str())), static_cast<real_t>(std::atof(vec[1].c_str()))));
			}
		}
		patterns.push_back(std::vector<cv::vec2r>());
	}

	stream.close();

	return patterns;
}

void pattern_detection(const std::vector<std::string> &calib_patterns, const std::string &out_path) {

	//std::string calib_patterns[] = {"1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png", "10.png"};

	std::vector<std::vector<cv::vec2r> > patterns;

	unsigned im_w = 0, im_h = 0;

	for (auto im_path : calib_patterns) {

		cv::matrixr image = cv::imread(/*"/home/relja/calibration/ground_truth/" + */im_path, cv::REAL, 1);

		im_w = image.cols();
		im_h = image.rows();

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

	if (write_pattern_results(patterns, im_w, im_h, out_path.c_str())) {
		std::cout << "Writing results to " << out_path << " successfull!" << std::endl;
	} else {
		std::cout << "Writing results to " << out_path << " failed!" << std::endl;
	}
}

bool read_zhang_data(const std::string &folderpath, std::vector<std::vector<cv::vec2r> > &image_points,
                     std::vector<cv::vec3r > &model_points)
{
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

	return true;
}

std::vector<std::vector<cv::vec2r> > image_all_pts;
std::vector<std::vector<cv::vec2r> > image_all_norm_pts;
std::vector<cv::matrixr> K_mats;
cv::vec2r *image_pts;
cv::vec3r *model_pts;
real_t *A_mat;
real_t *K_mat;
real_t *k_vec;

void ext_reprojection_fcn(int *m, int *n, double* x, double* fvec,int *iflag) {

	if (*iflag == 0)
		return;

	// calculate m_projected
	cv::matrixd A(3, 3, A_mat); // TODO: check if copies
	cv::matrixd K(3, 4, x);

	cv::matrixr ARt = A * K;

	cv::vectorr model(4);
	cv::vectorr res(3);
	cv::vectorr cam_ptn;

	cv::vec2r image_pt_proj;

	for (int i = 0; i < *m; ++i) {
		model[0] = model_pts[i][0];
		model[1] = model_pts[i][1];
		model[2] = 0.0;
		model[3] = 1.0;

		res = ARt * model;

		image_pt_proj[0] = res[0] / res[2];
		image_pt_proj[1] = res[1] / res[2];

		fvec[i] = sqrt(pow(image_pts[i][0] - image_pt_proj[0], 2) +
		            pow(image_pts[i][1] - image_pt_proj[1], 2));
	}
}

void distorion_reprojection_fcn(int *m, int *n, double* x, double* fvec,int *iflag) {

	if (*iflag == 0)
		return;

	// calculate m_projected
	cv::matrixd A(3, 3, A_mat); // TODO: check if copies
	cv::vectord k(x, x, 2, 1);

	cv::vectorr model(4);
	cv::vectorr res(3);
	cv::vectorr cam_ptn;

	cv::vec2r image_pt_proj;

	auto Uo =  A(0, 2);
	auto Vo =  A(1, 2);

	double x2_y2, u_uo, v_vo;

	unsigned f_i = 0;
	for (unsigned i = 0; i < image_all_pts.size(); ++i) {
		const cv::matrixd &K = K_mats[i];
		cv::matrixr ARt = A * K;
		for (unsigned j = 0; j < image_all_pts[i].size(); ++j) {
			
			model[0] = model_pts[j][0];
			model[1] = model_pts[j][1];
			model[2] = 0.0;
			model[3] = 1.0;

			res = ARt * model;

			image_pt_proj[0] = res[0] / res[2];
			image_pt_proj[1] = res[1] / res[2];

			x2_y2 = (image_all_norm_pts[i][j] * image_all_norm_pts[i][j]).sum();

			u_uo = image_pt_proj[0] - Uo;
			v_vo = image_pt_proj[1] - Vo;

			image_pt_proj[0] = image_pt_proj[0] + u_uo*(k[0]*(x2_y2) + k[1]*pow(x2_y2, 2));
			image_pt_proj[1] = image_pt_proj[1] + v_vo*(k[0]*(x2_y2) + k[1]*pow(x2_y2, 2));

			fvec[f_i++] = sqrt(pow(image_all_pts[i][j][0] - image_pt_proj[0], 2) +
					pow(image_all_pts[i][j][1] - image_pt_proj[1], 2));
		}
	}
}

void all_reprojection_fcn(int *m, int *n, double* x, double* fvec,int *iflag) {

	if (*iflag == 0) {
		return;
	}

	// calculate m_projected
	cv::matrixd A(3, 3, x); // TODO: check if copies
	cv::vectord k(x + *m - 2, x + *m - 1, 2, 1);

	cv::vectorr model(4);
	cv::vectorr res(3);
	cv::vectorr cam_ptn;

	cv::vec2r image_pt_proj;

	auto Uo =  A(0, 2);
	auto Vo =  A(1, 2);

	double x2_y2, u_uo, v_vo;

	unsigned f_i = 0;
	for (unsigned i = 0; i < image_all_pts.size(); ++i) {
		cv::matrixd K(3, 4, x + 9 + i*12);
		cv::matrixr ARt = A * K;
		for (unsigned j = 0; j < image_all_pts[i].size(); ++j) {
			
			model[0] = model_pts[j][0];
			model[1] = model_pts[j][1];
			model[2] = 0.0;
			model[3] = 1.0;

			res = ARt * model;

			image_pt_proj[0] = res[0] / res[2];
			image_pt_proj[1] = res[1] / res[2];

			x2_y2 = (image_all_norm_pts[i][j] * image_all_norm_pts[i][j]).sum();
			u_uo = image_pt_proj[0] - Uo;
			v_vo = image_pt_proj[1] - Vo;

			image_pt_proj[0] = image_pt_proj[0] + u_uo*(k[0]*(x2_y2) + k[1]*pow(x2_y2, 2));
			image_pt_proj[1] = image_pt_proj[1] + v_vo*(k[0]*(x2_y2) + k[1]*pow(x2_y2, 2));

			fvec[f_i++] = sqrt(pow(image_all_pts[i][j][0] - image_pt_proj[0], 2) +
					pow(image_all_pts[i][j][1] - image_pt_proj[1], 2));
		}
	}
}

int optimize_extrinsics(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, cv::matrixr &K, double tol = 1e-32) 
{
	ASSERT(image_points.size() == model_points.size() && !image_points.empty());

	image_pts = const_cast<cv::vec2r*>(image_points.data());
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());

	int m = image_points.size();
	int n = 12; // K.size

	int info = 0;

	auto *_K = new double[n];

	int i;
	for (i = 0; i < 12; ++i) {
		_K[i] = K.data_begin()[i];
	}

	info = cv::lmdif1(ext_reprojection_fcn, m, n, _K, tol);

	for (i = 0; i < 12; ++i) {
		K.data_begin()[i] = _K[i];
	}

	delete [] _K;

	return info;
}

int optimize_distortion(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<std::vector<cv::vec2r>> &image_points_nrm, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, const std::vector<cv::matrixr> &K, cv::vec2r &k, double tol = 1e-32) 
{

	image_all_pts = image_points;
	image_all_norm_pts = image_points_nrm;
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());
	K_mats = K;

	int m = image_points.size()*image_points[0].size();
	int n = 2;

	int info = 0;

	double data[2] = {k[0], k[1]};

	info = cv::lmdif(distorion_reprojection_fcn, m, n, data, 10000, 10e-08, 10e-08, 10e-08, 10e-64);

	k[0] = data[0];
	k[1] = data[1];

	return info;
}

int optimize_all(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<std::vector<cv::vec2r>> &image_points_nrm, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, std::vector<cv::matrixr> &K, cv::vec2r &k, double tol = 1e-32) 
{

	image_all_pts = image_points;
	image_all_norm_pts = image_points_nrm;
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());

	int m = image_points.size()*image_points[0].size();
	int n = 9 + (K.size()*12) + 2; // A + K + k;

	int info = 0;

	auto *data = new double[n];

	int d_i = 0;
	for (unsigned i = 0; i < 9; ++i, ++d_i) {
		data[d_i] = A.data_begin()[i];
	}
	for (unsigned b = 0; b < K.size(); ++b) {
		for (unsigned i = 0; i < 12; ++i, ++d_i) {
			data[d_i] = K[b].data_begin()[i];
		}
	}
	data[d_i++] = k[0];
	data[d_i++] = k[1];

	info = cv::lmdif(all_reprojection_fcn, m, n, data);

	d_i = 0;
	for (unsigned i = 0; i < 9; ++i, ++d_i) {
		A.data_begin()[i] = data[d_i];
	}
	for (unsigned b = 0; b < K.size(); ++b) {
		for (unsigned i = 0; i < 12; ++i, ++d_i) {
			K[b].data_begin()[i] = data[d_i];
		}
	}
	k[0] = data[d_i++];
	k[1] = data[d_i++];

	delete [] data;

	return info;
}

int main() {

	//pattern_detection();
	//return 1;

	unsigned im_w = 640, im_h = 480;

	std::vector<std::vector<cv::vec2r>> image_points_nrm;
	std::vector<cv::vec3r> model_points;

	read_zhang_data("/home/relja/git/camera_calibration/calib_data/zhang_data", image_points_nrm, model_points);

	auto image_points_orig = image_points_nrm;

	auto N = normalize_image_points(image_points_nrm, im_w, im_h);
	auto N_inv = N.clone(); 
	cv::invert(N_inv);

	auto image_points_count = image_points_nrm.size();

	std::vector<cv::matrixr> Hs(image_points_count);

	for (unsigned i = 0; i < image_points_count; ++i) {
		ASSERT(image_points_nrm[i].size() == model_points.size());

		Hs[i] = homography_solve(image_points_nrm[i], model_points);
		homography_optimize(image_points_nrm[i], model_points, Hs[i]);

		Hs[i] *= 1. / Hs[i](2, 2);

		std::cout << "Homography " << i << std::endl << Hs[i] << std::endl;
	}

	auto A_p = compute_intrisics(Hs);

	if (!A_p) {
		std::cout << "Failure calculating intrinsic parameters." << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "Intrinsics matrix A':" << std::endl;
	std::cout << A_p << std::endl;

	auto A = denormalize_intrinsics(A_p, N);
	std::cout << "Denormalized intrinsics matrix A:" << std::endl;
	std::cout << A << std::endl;

	std::vector<std::vector<cv::vec2r>> image_points_proj(image_points_count);
	std::vector<std::vector<cv::vec2r>> camera_points(image_points_count);

	std::vector<cv::matrixr> Ks;

	for (unsigned i = 0; i < image_points_count; ++i) {

		auto &H = Hs[i];

		auto K = compute_extrinsics(A, N_inv*H);

		std::cout << "Extrinsics " <<  i << std::endl;
		std::cout << "K non optimized:\n" << K << std::endl;
		std::cout << "A non optimized:\n" << A << std::endl;

		auto err = calc_reprojection_error(A, K, model_points, image_points_orig[i], image_points_proj[i], camera_points[i]);
		std::cout << "Non-optimized reprojection error: " << err << std::endl;

		optimize_extrinsics(image_points_orig[i], model_points, A, K);

		std::cout << "K:\n" << K << std::endl;
		std::cout << "A:\n" << A << std::endl;

		err = calc_reprojection_error(A, K, model_points, image_points_orig[i], image_points_proj[i], camera_points[i]);

		std::cout << "Reprojection error: " << err << std::endl;

		Ks.push_back(K);
	}

	auto k = compute_distortion(image_points_orig, image_points_nrm, image_points_proj, A);
	std::cout << "\nDistortion coefficients:\n" << k  << std::endl << std::endl;

	optimize_distortion(image_points_orig, image_points_nrm, model_points, A, Ks, k, 10e-128);

	std::cout << "After optimization:" << std::endl;
	std::cout << "k: " << k  << std::endl;

	optimize_all(image_points_orig, image_points_nrm, model_points, A, Ks, k, 10e-128);

	std::cout << "\n\n**********************************************************" << std::endl;
	std::cout << "A:\n" << A << std::endl;
	std::cout << "k:\n" << k << std::endl << std::endl;
	std::cout << "p:\n" << A(0, 2) << ", " << A(1, 2) << std::endl << std::endl;

	for (unsigned i = 0; i < image_points_count; ++i) {
		std::cout << "K" << i << ":\n" << Ks[i] << std::endl;
	}

	std::cout << "**********************************************************" << std::endl;

	return EXIT_SUCCESS;
}

