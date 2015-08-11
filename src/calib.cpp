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


#include "calib.hpp"


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

cv::matrixr denormalize_intrinsics(const cv::matrixr &A_p, const cv::matrixr &N) {
	auto N_inv = N.clone();
	cv::invert(N_inv);
	return (N_inv*A_p);
}

real_t calc_reprojection_error(const cv::matrixr &A, const cv::matrixr &K,
                               const std::vector<cv::vec3r> &model_pts, const std::vector<cv::vec2r> &image_pts,
                               std::vector<cv::vec2r> &image_pts_proj, std::vector<cv::vec2r> &camera_pts) {
	ASSERT(model_pts.size() == image_pts.size());

	auto m = model_pts.size();

	camera_pts = std::vector<cv::vec2r>(m);
	image_pts_proj = std::vector<cv::vec2r>(m);

	cv::vectorr model(4);
	cv::vectorr res(3);
	cv::vectorr cam_ptn(3);

	cv::matrixr ARt = A * K;

	real_t err = 0.0;

	for(unsigned i = 0; i < m; i++) {

		model[0] = model_pts[i][0];
		model[1] = model_pts[i][1];
		model[2] = 0.0;
		model[3] = 1.0;

		cam_ptn = K * model;
		res = ARt * model;

		camera_pts[i][0] = cam_ptn[0];
		camera_pts[i][1] = cam_ptn[1];
		image_pts_proj[i][0] = res[0] / res[2];
		image_pts_proj[i][1] = res[1] / res[2];

		err += sqrt(pow(image_pts[i][0] - image_pts_proj[i][0], 2) +
		            pow(image_pts[i][1] - image_pts_proj[i][1], 2));
	}

	return err / m;
}

cv::matrixr compute_intrisics(const std::vector<cv::matrixr> &Hs) {

	auto V = pack_v(Hs);
	auto b = solve_b(V);
	auto B = get_B_from_b(b);

	return get_intrinsic_mat(B);
}

cv::matrixr compute_extrinsics(const cv::matrixr &A, const cv::matrixr &H) {

	auto Ainv = A.clone();
	cv::invert(Ainv);

	cv::matrixr K(3, 4);

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

	auto t	= (Ainv * h3) * l3;

	cv::matrixr R = {
		{r1[0], r2[0], r3[0]},
		{r1[1], r2[1], r3[1]},
		{r1[2], r2[2], r3[2]}
	};

	cv::matrixr U, S, Vt;
	cv::svd_decomp(R, U, S, Vt);

	R = U * Vt;

	r1 = R.col(0);
	r2 = R.col(1);
	r3 = R.col(2);

	auto c1 = K.col(0);
	auto c2 = K.col(1);
	auto c3 = K.col(2);
	auto c4 = K.col(3);

	std::copy(r1.begin(), r1.end(), c1.begin());
	std::copy(r2.begin(), r2.end(), c2.begin());
	std::copy(r3.begin(), r3.end(), c3.begin());
	std::copy(t.begin(), t.end(), c4.begin());

	return K;
}

cv::vec2r compute_distortion(const std::vector<std::vector<cv::vec2r>> &image_pts, const std::vector<std::vector<cv::vec2r>> &image_pts_nrm,
                             const std::vector<std::vector<cv::vec2r>> &image_pts_proj, const cv::matrixr &A) {

	ASSERT(!image_pts.empty() && !image_pts_nrm.empty() && !image_pts_proj.empty());
	ASSERT(image_pts.front().size() == image_pts_nrm.front().size() &&
	       image_pts.front().size() == image_pts_proj.front().size());

	cv::vec2r k;
	real_t Uo,Vo,u_uo,v_vo, x2_y2;

	unsigned n_pts = image_pts.front().size();

	cv::matrixr D(image_pts.size()*n_pts*2, 2);
	cv::matrixr d(image_pts.size()*n_pts*2, 1);

	Uo =  A(0, 2);
	Vo =  A(1, 2);

	for (unsigned b = 0; b < image_pts.size(); ++b) {
		for (unsigned i = 0; i < n_pts; i++) {

			x2_y2 = (image_pts_nrm[b][i] * image_pts_nrm[b][i]).sum();
			u_uo = image_pts_proj[b][i][0] - Uo;
			v_vo = image_pts_proj[b][i][1] - Vo;

			D(i*2, 0) = (u_uo)*(x2_y2);
			D(i*2, 1) = (u_uo)*(x2_y2)*(x2_y2);

			D(i*2 + 1, 0) = (v_vo)*(x2_y2);
			D(i*2 + 1, 1) = (v_vo)*(x2_y2)*(x2_y2);

			d(i*2, 0) = image_pts[b][i][0] - image_pts_proj[b][i][0];
			d(i*2+1, 0) = image_pts[b][i][1] - image_pts_proj[b][i][1];
		}
	}

	cv::matrixr K;
	cv::lu_solve(D, d, K);

	k[0] = K.data()[0];
	k[1] = K.data()[1];

	return k;
}


