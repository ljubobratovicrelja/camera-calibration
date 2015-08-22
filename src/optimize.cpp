#include "optimize.hpp"

std::vector<std::vector<cv::vec2r> > image_all_pts;
std::vector<std::vector<cv::vec2r> > image_all_norm_pts;
std::vector<cv::matrixr> K_mats;
cv::vec2r *image_pts;
cv::vec3r *model_pts;
real_t *A_mat;
real_t *K_mat;
real_t *k_vec;

bool g_no_skew;
bool g_fixed_aspect;
unsigned a_param_count;

void ext_reprojection_fcn(int m, int n, real_t* x, real_t* fvec,int *iflag) {

	if (*iflag == 0)
		return;

	// calculate m_projected
	cv::matrixr A(3, 3, A_mat); 
	cv::matrixr K(3, 4, x);

	cv::vec2r image_pt_proj;

	for (int i = 0; i < m; ++i) {

		// pack model (world) 3D point.
		cv::vectorr model = { model_pts[i][0], model_pts[i][1], 0.0, 1.0};
		auto proj_ptn = (A*K) * model;
		proj_ptn /= proj_ptn[2];

		// calculate projection error
		auto x_d = image_pts[i][0] - proj_ptn[0];
		auto y_d = image_pts[i][1] - proj_ptn[1];

		x_d*=x_d;
		y_d*=y_d;

		fvec[i] = sqrt(x_d + y_d);
	}
}

int optimize_extrinsics(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, cv::matrixr &K, real_t tol) 
{
	ASSERT(image_points.size() == model_points.size() && !image_points.empty());

	image_pts = const_cast<cv::vec2r*>(image_points.data());
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());

	int m = image_points.size();
	int n = 12; // K.size
	int info;

	cv::vectorr _K(12);

	for (int i = 0; i < 12; ++i) {
		_K[i] = K.data_begin()[i];
	}

	if((info = cv::lmdif1(ext_reprojection_fcn, m, n, _K.data(), tol))) {
		for (int i = 0; i < 12; ++i) {
			K.data_begin()[i] = _K[i];
		}
	} else {
		std::cout << "Extrinsic optimization failed." << std::endl;
	}

	return info;
}

void pack_k_data(const cv::vectorr &k, real_t data[8]) {
	ASSERT(k.length() == 2 || k.length() == 4 || k.length() == 8);

	switch(k.length()) {
		case 2:
			data[0] = k[0]; 
			data[1] = k[1];
			data[2] = 0.; // k[2]
			data[3] = 0.; // k[3]
			data[4] = 0.; // k[4]
			data[5] = 0.; // k[5]
			data[6] = 0.; // p[0]
			data[7] = 0.; // p[1]
			break;
		case 4:
			data[0] = k[0]; 
			data[1] = k[1];
			data[2] = 0.; // k[2]
			data[3] = 0.; // k[3]
			data[4] = 0.; // k[4]
			data[5] = 0.; // k[5]
			data[6] = k[2]; // p[0]
			data[7] = k[3]; // p[1]
			break;
		case 8:
			data[0] = k[0]; 
			data[1] = k[1];
			data[2] = k[2]; // k[2]
			data[3] = k[3]; // k[3]
			data[4] = k[4]; // k[4]
			data[5] = k[5]; // k[5]
			data[6] = k[6]; // p[0]
			data[7] = k[7]; // p[1]
			break;
		default:
			break;
	}
}

cv::vectorr unpack_k_data(real_t data[8]) {

	cv::vectorr k(8);

	k[0] = data[0];
	k[1] = data[1];
	k[2] = data[2];
	k[3] = data[3];
	k[4] = data[4];
	k[5] = data[5];
	k[6] = data[6];
	k[7] = data[7];

	return k;
}

void distorion_reprojection_fcn(int m, int n, real_t* x, real_t* fvec,int *iflag) {

	if (*iflag == 0)
		return;

	// calculate m_projected
	cv::matrixr A(3, 3, A_mat);
	cv::vectorr k(x, x, 8, 1);

	unsigned f_i = 0;
	for (unsigned i = 0; i < image_all_pts.size(); ++i) {
		for (unsigned j = 0; j < image_all_pts[i].size(); ++j, ++f_i) {
			
			// pack model (world) 3D point.
			cv::vectorr model = { model_pts[j][0], model_pts[j][1], 0.0, 1.0};
			auto proj_ptn = reproject_point(model, A, K_mats[i], k);

			// calculate projection error
			auto x_d = image_all_pts[i][j][0] - proj_ptn[0];
			auto y_d = image_all_pts[i][j][1] - proj_ptn[1];

			x_d*=x_d;
			y_d*=y_d;

			fvec[f_i] = sqrt(x_d + y_d);
		}
	}
}

int optimize_distortion(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		const cv::matrixr &A, const std::vector<cv::matrixr> &K, cv::vectorr &k, real_t tol) 
{

	ASSERT(k.length() == 2 || k.length() == 4 || k.length() == 8);

	image_all_pts = image_points;
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());
	K_mats = K;

	int m = image_points.size()*image_points[0].size();
	int n = 2;

	int info = 0;

	real_t data[8];
	pack_k_data(k, data);

	if((info = cv::lmdif1(distorion_reprojection_fcn, m, n, data, tol))) {
		k = unpack_k_data(data);
	} else {
		std::cout << "Distortion optimization did not converge" << std::endl;
	}

	return info;
}

cv::matrixr construct_a(real_t *x) {

	cv::matrixr A(3, 3);

	if (g_fixed_aspect) {

		if (g_no_skew) {
			A(0, 0) = x[0];
			A(1, 1) = x[0];
			A(0, 2) = x[1];
			A(1, 2) = x[2];
			A(0, 1) = 0;
		} else {
			A(0, 0) = x[0];
			A(1, 1) = x[0];
			A(0, 1) = x[1];
			A(0, 2) = x[2];
			A(1, 2) = x[3];
		}
	} else {
		if (g_no_skew) {
			A(0, 0) = x[0];
			A(0, 2) = x[1];
			A(1, 1) = x[2];
			A(1, 2) = x[3];
			A(0, 1) = 0;
		} else {
			A(0, 0) = x[0];
			A(0, 1) = x[1];
			A(0, 2) = x[2];
			A(1, 1) = x[3];
			A(1, 2) = x[4];
		}
	}

	A(1, 0) = A(2, 0) = A(2, 1) = 0;
	A(2, 2) = 1;

	return A;
}

void all_reprojection_fcn(int m, int n, real_t* x, real_t* fvec,int *iflag) {

	if (*iflag == 0) {
		return;
	}

	// calculate m_projected
	auto A = construct_a(x);

	auto k_x = x + n - 8;
	cv::vectorr k(k_x, k_x, 8, 1);

	unsigned f_i = 0;

	for (unsigned i = 0; i < image_all_pts.size(); ++i) {
		cv::matrixr K(3, 4, (x + a_param_count + (i * 12)));
		for (unsigned j = 0; j < image_all_pts[i].size(); ++j, ++f_i) {
			
			// pack model (world) 3D point.
			cv::vectorr model = { model_pts[j][0], model_pts[j][1], 0.0, 1.0};
			auto proj_ptn = reproject_point(model, A, K, k);

			// calculate projection error
			auto x_d = image_all_pts[i][j][0] - proj_ptn[0];
			auto y_d = image_all_pts[i][j][1] - proj_ptn[1];

			x_d*=x_d;
			y_d*=y_d;

			fvec[f_i] = sqrt(x_d + y_d);
		}
	}
}

int optimize_calib(const std::vector<std::vector<cv::vec2r>> &image_points, const std::vector<cv::vec3r> &model_points, 
		cv::matrixr &A, std::vector<cv::matrixr> &K, cv::vectorr &k, bool fixed_aspect, bool no_skew, real_t tol) 
{

	image_all_pts = image_points;
	model_pts = const_cast<cv::vec3r*>(model_points.data());
	A_mat = const_cast<real_t*>(A.data_begin());

	g_fixed_aspect = fixed_aspect;
	g_no_skew = no_skew;

	a_param_count = fixed_aspect ? 3 : 4;
	a_param_count += no_skew ? 0 : 1;

	int m = image_points.size()*image_points[0].size();
	int n = a_param_count + (K.size()*12) + 8; // A{a, b, c, u0, v0} + K + k;

	int info = 0;

	auto *data = new real_t[n];

	if (fixed_aspect) {
		if (no_skew) {
			data[0] = (A(0, 0) + A(1, 1)) / 2;
			data[1] = A(0, 2);
			data[2] = A(1, 2);
		} else {
			data[0] = (A(0, 0) + A(1, 1)) / 2;
			data[1] = A(0, 1);
			data[2] = A(0, 2);
			data[3] = A(1, 2);
		}
	} else {
		if (no_skew) {
			data[0] = A(0, 0);
			data[1] = A(0, 2);
			data[2] = A(1, 1);
			data[3] = A(1, 2);
		} else {
			data[0] = A(0, 0);
			data[1] = A(0, 1);
			data[2] = A(0, 2);
			data[3] = A(1, 1);
			data[4] = A(1, 2);
		}
	}

	for (unsigned b = 0; b < K.size(); ++b) {
		for (unsigned i = 0; i < 12; ++i) {
			data[a_param_count + (b * 12) + i] = K[b].data_begin()[i];
		}
	}

	auto k_str = data + (n - 8);
	pack_k_data(k, k_str);

	if((info = cv::lmdif1(all_reprojection_fcn, m, n, data, tol))) {
		A = construct_a(data);
		for (unsigned b = 0; b < K.size(); ++b) {
			cv::matrixr K_(3, 4, (data + a_param_count + (b * 12)));
			K[b] = K_.clone();
		}
		k = unpack_k_data(k_str);
	} else {
		std::cout << "Optimization failed." << std::endl;
	}

	delete [] data;

	return info;
}
