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
#include <matfunc.hpp>

#include <iostream>
#include <fstream>

#include <minpack.h>

#include "calibpattern.hpp"
#include "homography.hpp"

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

cv::matrixr normalize_image_points(std::vector<std::vector<cv::vec2r> > &patterns, unsigned w, unsigned h) {

	real_t sx = 2. / w;
	real_t sy = 2. / h;
	real_t x0 = w / 2.;
	real_t y0 = h / 2.;

	for(unsigned i = 0; i < patterns.size(); ++i) {
		for(unsigned j = 0; j < patterns[i].size(); ++j) {
			patterns[i][j][0] = sx*(patterns[i][j][0]-x0);
			patterns[i][j][1] = sy*(patterns[i][j][1]-y0);
		}
	}

	return {
		{sx, 0.,  -sx*x0},
		{0., sy, -sy*y0},
		{0., 0., 1.}
	};
}

cv::matrixr denormalize_intrinsics(cv::matrixr &A_p, const cv::matrixr &N) {
	auto N_inv = N.clone();
	cv::invert(N_inv);
	return (N_inv*A_p);
}

void pattern_detection() {

	std::string calib_patterns[] = {"1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png", "10.png"};

	std::vector<std::vector<cv::vec2r> > patterns;

	unsigned im_w = 0, im_h = 0;

	for (auto im_path : calib_patterns) {

		cv::matrixr image = cv::imread("/home/relja/calibration/ground_truth/" + im_path, cv::REAL, 1);

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

	std::string out_path;
	std::cout << "Where to write results?" << std::endl;
	std::cin >> out_path;

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

std::vector<cv::vec3r> calculate_object_points(unsigned rows, unsigned cols, real_t square_size) {
	std::vector<cv::vec3r> obj_pts;
	obj_pts.reserve(rows*cols);

	for(unsigned i = 0; i < rows; ++i) {
		for(unsigned j = 0; j < cols; ++j) {
			obj_pts.push_back({static_cast<real_t>(j*square_size), static_cast<real_t>(i*square_size), 1.});
		}
	}

	return obj_pts;
}

real_t calc_reprojection_error(const cv::matrixr &H, const std::vector<cv::vec2r> &source_pts, const std::vector<cv::vec3r> &target_pts) {

	ASSERT(source_pts.size() == target_pts.size() && H && H.rows() == 3 && H.cols() == 3);

	unsigned ptn_count = source_pts.size();
	real_t err = 0.0;

	// calculate m_projected
	cv::matrixr ptn(3, 1), p_ptn(3, 1), res_ptn(3, 1);

	for (unsigned i = 0; i < ptn_count; ++i) {
		ptn(0, 0) = source_pts[i][0];
		ptn(1, 0) = source_pts[i][1];
		ptn(2, 0) = 1.;

		p_ptn(0, 0) = target_pts[i][0];
		p_ptn(1, 0) = target_pts[i][1];
		p_ptn(2, 0) = target_pts[i][2];

		cv::cross( H, ptn, res_ptn);

		res_ptn(0, 0) /= res_ptn(2, 0);
		res_ptn(1, 0) /= res_ptn(2, 0);
		res_ptn(2, 0) = 1;

		err += pow(cv::distance(res_ptn, p_ptn, cv::Norm::L2), 2);
	}

	return sqrt(err);
}

cv::vectorr get_vij(const cv::matrixr &h, unsigned i, unsigned j) {
	return cv::vectorr {
		h(0, i)*h(0, j), h(0, i)*h(1, j) + h(1, i)*h(0, j), h(1, i)*h(1, j),
		h(2, i)*h(0, j) + h(0, i)*h(2, j), h(2, i)*h(1, j) + h(1, i)*h(2, j), h(2, i)*h(2, j)
	};
}

cv::matrixr pack_v(const std::vector<cv::matrixr> &Hs) {
	cv::matrixr v(2*Hs.size(), 6);

	for (unsigned i = 0; i < Hs.size(); ++i) {

		auto h_r_1 = v.row(i*2);
		auto h_r_2 = v.row(i*2 + 1);

		auto v12 = get_vij(Hs[i], 0, 1);
		auto v11 = get_vij(Hs[i], 0, 0);
		auto v22 = get_vij(Hs[i], 1, 1);
		auto v11_v22 = v11 - v22;

		std::copy(v12.begin(),v12.end(),h_r_1.begin());
		std::copy(v11_v22.begin(), v11_v22.end(), h_r_2.begin());
	}

	return v;
}

cv::vectorr solve_b(const cv::matrixr &V) {

	cv::matrixr U, S, Vt;
	cv::svd_decomp(V, U, S, Vt);

	real_t smallest_sv = std::numeric_limits<real_t>::max();
	unsigned smallest_i = 0;
	for (unsigned i = 0; i < S.rows(); ++i) {
		if (S(i, i) < smallest_sv) {
			smallest_sv = S(i, i);
			smallest_i = i;
		}
	}

	return Vt.transposed().col(smallest_i);
}

cv::matrixr get_B_from_b(const cv::vectorr &b) {
	return {
		{b[0], b[1], b[3]},
		{b[1], b[2], b[4]},
		{b[3], b[4], b[5]}
	};
}

bool extract_intrinsics_from_B(const cv::matrixr &B, real_t &u0, real_t &v0,
                               real_t &lambda, real_t &alpha, real_t &beta, real_t &gama) {
	auto den = B(0, 0) * B(2, 2) - B(0, 1)*B(0, 1);

	if (fabs(den) < 1e-8) {
		std::cout << "Den < 1e-8" << std::endl;
		return false;
	}

	v0 = (B(0, 1)*B(0, 2) - B(0, 0)*B(1, 2)) / (B(0, 0)*B(1, 1) - B(0, 1)*B(0, 1));
	lambda = B(2, 2) - (B(0, 1)*B(0, 1) + v0*(B(0, 1)*B(0, 2) - B(0, 0)*B(1, 2))) / B(0, 0);
	auto l = (lambda / B(0, 0));
	if (l < .0) {
		std::cout << "L < 0: " << l << std::endl;
		return false;
	}
	alpha = sqrt(l);
	auto b =(lambda*B(0, 0))/(B(0, 0)*B(1, 1) - B(0, 1)*B(0, 1));
	if (b < .0) {
		std::cout << "beta < 0: " << b << std::endl;
		return false;
	}
	beta = sqrt(b);
	gama = (-1*B(0, 1)*(alpha*alpha)*beta)/lambda;
	u0 = (gama*v0)/alpha - (B(0, 2)*(alpha*alpha))/lambda;

	return true;
}

cv::matrixr get_intrinsic_mat(const cv::matrixr &B) {

	real_t u0, v0, a, b, c, lambda;
	if (extract_intrinsics_from_B(B, u0, v0, lambda, a, b, c)) {
		return {
			{a, c, u0},
			{0., b, v0},
			{0., 0., 1.}
		};
	} else {
		std::cerr << "Failure calculation A'" << std::endl;
		return cv::matrixr();
	}
}

void m3MultAb(real_t* A, real_t* b, real_t* x) {
   x[0]=A[0]*b[0]+A[1]*b[1]+A[2]*b[2];
   x[1]=A[3]*b[0]+A[4]*b[1]+A[5]*b[2];
   x[2]=A[6]*b[0]+A[7]*b[1]+A[8]*b[2];
}

void compute_extrinsics(cv::matrixr Ainv, cv::matrixr H, cv::matrixr &R, cv::vec3r &t) {

	auto h1 = H.col(0);
	auto h2 = H.col(1);
	auto h3 = H.col(2);

	auto r1 = Ainv * h1;
	auto r2 = Ainv * h2;

	real_t l1 = 1. / r1.norm();
	real_t l2 = 1. / r2.norm();
	real_t l3 = (l1 + l2) / 2.;

	r1.normalize();
	r2.normalize();
	auto r3 = r1.cross(r2);

	auto tv = (Ainv * h3) * l3;

	t[0] = tv[0];
	t[1] = tv[1];
	t[2] = tv[2];

	R.create(3, 3);
	auto c1 = R.col(0);
	auto c2 = R.col(1);
	auto c3 = R.col(2);

	std::copy(r1.begin(), r1.end(), c1.begin());
	std::copy(r2.begin(), r2.end(), c2.begin());
	std::copy(r3.begin(), r3.end(), c3.begin());
}

int main() {

	//pattern_detection();
	//return 1;

	unsigned im_w = 640, im_h = 480;

	std::vector<std::vector<cv::vec2r>> image_points;
	std::vector<cv::vec3r> model_points;

	read_zhang_data("/home/relja/git/camera_calibration/calib_data/zhang_data", image_points, model_points);

	auto N = normalize_image_points(image_points, im_w, im_h);
	auto N_inv = N.clone(); 
	cv::invert(N_inv);

	auto image_points_count = image_points.size();

	std::vector<cv::matrixr> Hs(image_points_count);

	for (unsigned i = 0; i < image_points_count; ++i) {
		ASSERT(image_points[i].size() == model_points.size());

		auto &H = Hs[i];

		homography_solve(image_points[i], model_points, H);

		homography_optimization::source_pts = image_points[i];
		homography_optimization::target_pts = model_points;
		homography_optimization::evaluate(H, homography_optimization::reprojection_fcn);

		H *= 1. / H(2, 2);

		std::cout << "Homography " << i << std::endl << H << std::endl;
	}

	auto V = pack_v(Hs);
	std::cout << "V:\n" << V << std::endl;
	auto b = solve_b(V);
	auto B = get_B_from_b(b);

	std::cout << "B matrix:" << std::endl;
	std::cout << B << std::endl;

	auto A_p = get_intrinsic_mat(B);

	if (!A_p) {
		std::cout << "Failure calculating intrinsic parameters." << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "Intrinsics matrix A':" << std::endl;
	std::cout << A_p << std::endl;

	auto A = denormalize_intrinsics(A_p, N);
	std::cout << "Denormalized intrinsics matrix A:" << std::endl;
	std::cout << A << std::endl;

	auto A_inv = A.clone();
	cv::invert(A_inv);

	int i = 0;
	for (auto H : Hs) {
		cv::matrixr R;
		cv::vec3r t; // extrinsics

		compute_extrinsics(A_inv, N_inv*H, R, t);
		std::cout << "Extrinsics " <<  (i++) << std::endl;
		std::cout << "R:\n" << R << std::endl;
		std::cout << "t:\n" << t << std::endl;
		std::cout << std::endl;
	}

	return EXIT_SUCCESS;
}







