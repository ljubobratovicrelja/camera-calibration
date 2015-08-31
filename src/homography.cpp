#include "homography.hpp"

#include <linalg.hpp>
#include <math.hpp>


cv::matrixr homography_8_point(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points) {

	cv::matrixr H;

	auto n = image_points.size();
	ASSERT(model_points.size() == n);
	cv::matrixr L = cv::matrixr::zeros(2*n, 9);

	for(unsigned k = 0; k < n; k++) {

		real_t X=model_points[k][0];  /* X coord of model point k */
		real_t Y=model_points[k][1];  /* Y coord of model point k */
		real_t W=model_points[k][2];  /* W coord of model point k */
		real_t u=image_points[k][0];  /* u coord of image point k */
		real_t v=image_points[k][1];  /* v coord of image point k */

		int i = 2*k;                 /* line number in matrix L  */

		L(i,0) =    X;
		L(i, 1) =    Y;
		L(i, 2) =    W;
		L(i, 3) =    0;
		L(i, 4) =    0;
		L(i, 5) =    0;
		L(i, 6) = -u*X;
		L(i, 7) = -u*Y;
		L(i, 8) = -u*W;

		i++;

		L(i, 0) =    0;
		L(i, 1) =    0;
		L(i, 2) =    0;
		L(i, 3) =    X;
		L(i, 4) =    Y;
		L(i, 5) =    W;
		L(i, 6) = -v*X;
		L(i, 7) = -v*Y;
		L(i, 8) = -v*W;
	}

	cv::null_solve(L, H);
	H.reshape(3, 3);

	H *= 1. / H(2, 2);

	return H;
}

// Similarity estimation for normalization process.
template<size_t _size>
cv::matrixr homography_dlt_sim_estimation(const std::vector<cv::vectorx<real_t, _size> > &features) {
	cv::matrixr transform = cv::matrixr::eye(3);

	cv::vec2r centroid(0, 0);
	cv::matrixr S;

	for (auto feat : features) {
		centroid += feat;
	}
	centroid /= features.size();

	real_t sum_dist = 0;

	for (auto feat : features) {
		sum_dist+= centroid.distance(feat);
	}
	centroid *= -1;

	real_t scale_v = std::sqrt(2.) / (sum_dist / features.size());

	transform(0, 0) = scale_v;
	transform(1, 1) = scale_v;
	transform(0, 2) = centroid[0];
	transform(1, 2) = centroid[1];

	return transform;
}

template<size_t _size>
void homography_dlt_normalize(std::vector<cv::vectorx<real_t, _size> > &features, const cv::matrixr &S) {
	ASSERT(S && S.rows() == 3 && S.cols() == 3);
	cv::matrixr x(3, 1), xp(3, 1);
	for (unsigned i = 0; i < features.size(); ++i) {
		x(0, 0) = features[i][0];
		x(1, 0) = features[i][1];
		x(2, 0) = (_size == 3) ? features[i][2] : 1.;
		cross(S, x, xp);
		features[i][0] = xp(0, 0) / xp(2, 0);
		features[i][1] = xp(1, 0) / xp(2, 0);
		if(_size == 3)
			features[i][2] = 1.;
	}
}

cv::matrixr homography_dlt(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec3r> &tgt_pts) {
	ASSERT(src_pts.size() >= 4 && src_pts.size() == tgt_pts.size());

	cv::matrixr H;

	// 0. Prepare data;
	cv::matrixr srcS, tgtS, invTgtS;
	cv::matrixr A = cv::matrixr::zeros(2 * src_pts.size(), 9);

	// 1. Perform normalization;
	srcS = homography_dlt_sim_estimation<2>(src_pts);
	tgtS = homography_dlt_sim_estimation<3>(tgt_pts);

	auto src_n = src_pts; // source normalized points
	auto tgt_n = tgt_pts; // target normalized points

	invTgtS = tgtS.clone();
	invert(invTgtS);

	homography_dlt_normalize<2>(src_n, srcS);
	homography_dlt_normalize<3>(tgt_n, tgtS);

	// 2. Pack matrix A;
	for (unsigned i = 0; i < src_pts.size(); ++i) {
		A(i * 2 + 0, 0) = -1 * src_n[i][0];
		A(i * 2 + 0, 1) = -1 * src_n[i][1];
		A(i * 2 + 0, 2) = -1;
		A(i * 2 + 0, 6) = tgt_n[i][0] * src_n[i][0];
		A(i * 2 + 0, 7) = tgt_n[i][0] * src_n[i][1];
		A(i * 2 + 0, 8) = tgt_n[i][0];

		A(i * 2 + 1, 3) = -1 * src_n[i][0];
		A(i * 2 + 1, 4) = -1 * src_n[i][1];
		A(i * 2 + 1, 5) = -1;
		A(i * 2 + 1, 6) = tgt_n[i][1] * src_n[i][0];
		A(i * 2 + 1, 7) = tgt_n[i][1] * src_n[i][1];
		A(i * 2 + 1, 8) = tgt_n[i][1];
	}

	// 3. solve nullspace of A for H;
	cv::null_solve(A, H);

	H.reshape(3, 3);

	// 4. denormalize the homography.
	H = invTgtS * H * srcS;

	return H;
}

/*
 * Pack homography matrices A and B by the form used for least squares solving.
 */
void pack_ab(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec3r> &tgt_pts, cv::matrixr &A, cv::matrixr &B) {

	ASSERT(src_pts.size() && src_pts.size() == tgt_pts.size());

	// construct matrices
	A = cv::matrixr::zeros(src_pts.size() * 2, 8);
	B.create(src_pts.size() * 2, 1);

	// populate matrices with data.
	for (unsigned i = 0; i < src_pts.size(); i++) {

		auto &src = src_pts[i];
		auto &tgt = tgt_pts[i];

		B(i * 2, 0) = tgt[0];
		B(i * 2 + 1, 0) = tgt[1];

		A(i * 2, 0) = src[0];
		A(i * 2, 1) = src[1];
		A(i * 2, 2) = 1;
		A(i * 2 + 1, 3) = src[0];
		A(i * 2 + 1, 4) = src[1];
		A(i * 2 + 1, 5) = 1;

		A(i * 2, 6) = -1 * src[0] * tgt[0];
		A(i * 2, 7) = -1 * src[1] * tgt[0];
		A(i * 2 + 1, 6) = -1 * src[0] * tgt[1];
		A(i * 2 + 1, 7) = -1 * src[1] * tgt[1];
	}
}

/*
 * Solve homography using least squares method.
 */
cv::matrixr homography_least_squares(const std::vector<cv::vec2r> &src_pts, const std::vector<cv::vec3r> &tgt_pts) {

	cv::matrixr A, B, H;
	pack_ab(src_pts, tgt_pts, A, B);
	cv::matrixr _H(8, 1);

	cv::matrixr At = A.transposed();
	lu_solve(At * A, At * B, _H);

	if (!_H) {
		throw std::runtime_error("Internal error!~ failure occurred calculating homography.\n");
	}

	H.create(1, 9);
	std::copy(_H.begin(), _H.end(), H.begin());
	H.reshape(3, 3);
	H(2, 2) = 1;

	return H;
}

cv::matrixr homography_solve(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points, H_calc_alg alg) {
	switch (alg) {
	case HOMOGRAPHY_8_POINT:
		return homography_8_point(image_points, model_points);
	case HOMOGRAPHY_LEAST_SQUARES:
		return homography_least_squares(image_points, model_points);
	case HOMOGRAPHY_DLT:
		return homography_dlt(image_points, model_points);
	};
}

std::vector<cv::vec2r> source_pts;
std::vector<cv::vec3r> target_pts;

void reprojection_fcn(int m, int n, real_t* x, real_t* fvec,int *iflag) {

	if (*iflag == 0)
		return;

	// calculate m_projected
	cv::matrixr _H(3, 3, x); // borrow x and form matrix
	cv::matrixr ptn(3, 1), p_ptn(3, 1), res_ptn(3, 1);

	for (int i = 0; i < m; ++i) {
		ptn(0, 0) = target_pts[i][0]; // model point
		ptn(1, 0) = target_pts[i][1];
		ptn(2, 0) = target_pts[i][2];

		p_ptn(0, 0) = source_pts[i][0]; // photo projection point
		p_ptn(1, 0) = source_pts[i][1];
		p_ptn(2, 0) = 1.;

		cv::cross( _H, ptn, res_ptn);

		res_ptn(0, 0) /= res_ptn(2, 0);
		res_ptn(1, 0) /= res_ptn(2, 0);
		res_ptn(2, 0) = 1.;

		fvec[i] = sqrt(pow(p_ptn(0, 0) - res_ptn(0, 0), 2) + pow(p_ptn(1, 0) - res_ptn(1, 0), 2));
	}
}

int homography_optimize(const std::vector<cv::vec2r> &image_points, const std::vector<cv::vec3r> &model_points,
                        cv::matrixr &H, real_t tol) {

	source_pts = image_points;
	target_pts = model_points;

	ASSERT(source_pts.size() > 9);

	int m = source_pts.size();
	int n = 9;

	int info = 0;

	auto *_H = new real_t[n];

	for (int i = 0; i < 9; ++i) {
		_H[i] = H.data_begin()[i];
	}

	info = cv::lmdif1(reprojection_fcn, m, n, _H, tol);

	for (int i = 0; i < 9; ++i) {
		H.data_begin()[i] = _H[i];
	}

	H /= H(2, 2);

	delete [] _H;

	return info;
}

real_t calc_h_reprojection_error(const cv::matrixr &H, const std::vector<cv::vec2r> &source_pts, const std::vector<cv::vec3r> &target_pts) {

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

		err += cv::distance(res_ptn, p_ptn, cv::Norm::L2);
	}

	return err / ptn_count;
}




