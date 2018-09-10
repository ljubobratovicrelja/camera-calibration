#include "optimize.hpp"

#include <cassert>

#include <cminpack.h>


std::vector<std::vector<cv::vec2r> > image_all_pts;
std::vector<std::vector<cv::vec2r> > image_all_norm_pts;
std::vector<cv::matrixr> K_mats;
cv::vec2r* image_pts;
cv::vec3r* model_pts;
real_t* A_mat;
real_t* K_mat;
real_t* k_vec;

bool g_no_skew;
bool g_fixed_aspect;
unsigned a_param_count;
unsigned k_param_count;


void ext_reprojection_fcn(int m, int n, real_t* x, real_t* fvec, int* iflag) {
    if (*iflag == 0) {
        return;
    }

    // calculate m_projected
    cv::matrixr A(3, 3, A_mat);
    cv::matrixr K(3, 4, x);

    cv::vec2r image_pt_proj;

    for (int i = 0; i < m; ++i) {
        // pack model (world) 3D point.
        cv::vectorr model = {model_pts[i][0], model_pts[i][1], 0.0, 1.0};
        auto proj_ptn = (A * K) * model;
        proj_ptn /= proj_ptn[2];

        // calculate projection error
        auto x_d = image_pts[i][0] - proj_ptn[0];
        auto y_d = image_pts[i][1] - proj_ptn[1];

        x_d *= x_d;
        y_d *= y_d;

        fvec[i] = sqrt(x_d + y_d);
    }
}

int optimize_extrinsics(const std::vector<cv::vec2r>& image_points,
                        const std::vector<cv::vec3r>& model_points,
                        const cv::matrixr& A,
                        cv::matrixr& K,
                        real_t tol) {
    ASSERT(image_points.size() == model_points.size() && !image_points.empty());

    image_pts = const_cast<cv::vec2r*>(image_points.data());
    model_pts = const_cast<cv::vec3r*>(model_points.data());
    A_mat = const_cast<real_t*>(A.data_begin());

    int m = image_points.size();
    int n = 12;  // K.size
    int info;

    cv::vectorr _K(12);

    for (int i = 0; i < 12; ++i) {
        _K[i] = K.data_begin()[i];
    }

    if ((info = cv::lmdif1(ext_reprojection_fcn, m, n, _K.data(), tol))) {
        for (int i = 0; i < 12; ++i) {
            K.data_begin()[i] = _K[i];
        }
    } else {
        std::cout << "Extrinsic optimization failed." << std::endl;
    }

    return info;
}

inline void pack_k_data(const cv::vectorr& k, real_t data[8]) {
    k_param_count = k.length();
    switch (k.length()) {
        case 2:
            data[0] = k[0];
            data[1] = k[1];
            break;
        case 4:
            data[0] = k[0];
            data[1] = k[1];
            data[2] = k[2];  // p[0]
            data[3] = k[3];  // p[1]
            break;
        case 8:
            data[0] = k[0];
            data[1] = k[1];
            data[2] = k[2];  // k[2]
            data[3] = k[3];  // k[3]
            data[4] = k[4];  // k[4]
            data[5] = k[5];  // k[5]
            data[6] = k[6];  // p[0]
            data[7] = k[7];  // p[1]
            break;
        default:
            break;
    }
}

cv::vectorr unpack_k_data(real_t* data) {
    if (k_param_count) {
        cv::vectorr k(k_param_count);
        switch (k.length()) {
            case 2:
                k[0] = data[0];
                k[1] = data[1];
                break;
            case 4:
                k[0] = data[0];
                k[1] = data[1];
                k[2] = data[2];
                k[3] = data[3];
                break;
            case 8:
                k[0] = data[0];
                k[1] = data[1];
                k[2] = data[2];
                k[3] = data[3];
                k[4] = data[4];
                k[5] = data[5];
                k[6] = data[6];
                k[7] = data[7];
                break;
            default:
                break;
        }

        return k;
    } else {
        return cv::vectorr();
    }
}

void distorion_reprojection_fcn(int m, int n, real_t* x, real_t* fvec, int* iflag) {
    if (*iflag == 0) {
        return;
    }

    // calculate m_projected
    cv::matrixr A(3, 3, A_mat);
    cv::vectorr k(x, x, 8, 1);

    unsigned f_i = 0;
    for (unsigned i = 0; i < image_all_pts.size(); ++i) {
        for (unsigned j = 0; j < image_all_pts[i].size(); ++j, ++f_i) {
            // pack model (world) 3D point.
            cv::vectorr model = {model_pts[j][0], model_pts[j][1], 0.0, 1.0};
            auto proj_ptn = reproject_point(model, image_all_norm_pts[i][j], A, K_mats[i], k);

            // calculate projection error
            auto x_d = image_all_pts[i][j][0] - proj_ptn[0];
            auto y_d = image_all_pts[i][j][1] - proj_ptn[1];

            x_d *= x_d;
            y_d *= y_d;

            fvec[f_i] = sqrt(x_d + y_d);
        }
    }
}

int optimize_distortion(const std::vector<std::vector<cv::vec2r> >& image_points,
                        const std::vector<std::vector<cv::vec2r> >& image_points_nrm,
                        const std::vector<cv::vec3r>& model_points,
                        const cv::matrixr& A,
                        const std::vector<cv::matrixr>& K,
                        cv::vectorr& k,
                        real_t tol) {
    ASSERT(k.length() == 2 || k.length() == 4 || k.length() == 8);

    image_all_pts = image_points;
    image_all_norm_pts = image_points_nrm;
    model_pts = const_cast<cv::vec3r*>(model_points.data());
    A_mat = const_cast<real_t*>(A.data_begin());
    K_mats = K;

    int m = image_points.size() * image_points[0].size();
    int n = 2;

    int info = 0;

    real_t data[8];
    pack_k_data(k, data);

    if ((info = cv::lmdif1(distorion_reprojection_fcn, m, n, data, tol))) {
        k = unpack_k_data(data);
    } else {
        std::cout << "Distortion optimization did not converge" << std::endl;
    }

    return info;
}

cv::matrixr construct_a(real_t* x) {
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

void all_reprojection_fcn(int m, int n, real_t* x, real_t* fvec, int* iflag) {
    if (*iflag == 0) {
        return;
    }

    // calculate m_projected
    auto A = construct_a(x);

    auto k_x = x + n - k_param_count;
    cv::vectorr k;

    if (k_param_count) {
        k = cv::vectorr(k_x, k_x, k_param_count, 1);
    }

    unsigned f_i = 0;
    for (unsigned i = 0; i < image_all_pts.size(); ++i) {
        cv::matrixr K(3, 4, (x + a_param_count + (i * 12)));
        auto image_pts = image_all_pts[i].size();
        for (unsigned j = 0; j < image_pts; ++j) {
            // pack model (world) 3D point.
            cv::vectorr model = {model_pts[j][0], model_pts[j][1], 0.0, 1.0};
            auto proj_ptn = reproject_point(model, image_all_norm_pts[i][j], A, K, k);

            // calculate projection error
            auto x_d = image_all_pts[i][j][0] - proj_ptn[0];
            auto y_d = image_all_pts[i][j][1] - proj_ptn[1];

            fvec[f_i++] = std::hypot(x_d, y_d);
        }
    }
}

/*
int optimize_calib(const std::vector<std::vector<cv::vec2r> >& image_points,
                   const std::vector<std::vector<cv::vec2r> >& image_points_nrm,
                   const std::vector<cv::vec3r>& model_points,
                   cv::matrixr& A,
                   std::vector<cv::matrixr>& K,
                   cv::vectorr& k,
                   bool fixed_aspect,
                   bool no_skew,
                   real_t tol) {
    image_all_pts = image_points;
    image_all_norm_pts = image_points_nrm;
    model_pts = const_cast<cv::vec3r*>(model_points.data());
    A_mat = const_cast<real_t*>(A.data_begin());
    K_mats = K;

    g_fixed_aspect = fixed_aspect;
    g_no_skew = no_skew;

    a_param_count = fixed_aspect ? 3 : 4;
    a_param_count += no_skew ? 0 : 1;

    int m = image_points.size() * image_points[0].size();
    int n = a_param_count + (K.size() * 12) + k.length();  // A{a, b, c, u0, v0} + K + k;
    int info = 0;

    auto* data = new real_t[n];

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

    auto k_str = data + (n - k.length());
    pack_k_data(k, k_str);

    if ((info = cv::lmdif1(all_reprojection_fcn, m, n, data, tol))) {
        A = construct_a(data);
        for (unsigned b = 0; b < K.size(); ++b) {
            cv::matrixr K_(3, 4, (data + a_param_count + (b * 12)));
            K[b] = K_.clone();
        }
        k = unpack_k_data(k_str);
    } else {
        std::cout << "Optimization failed." << std::endl;
    }

    delete[] data;

    return info;
}
*/

struct CalibrationOptimizationData {
    const std::vector<std::vector<cv::vec2r> >& target_pts;
    const std::vector<std::vector<cv::vec2r> >& target_pts_nrm;
    const std::vector<cv::vec3r>& model_pts;
    const size_t board_count;
};


std::vector<real_t> setupX(cv::matrixr& A, std::vector<cv::matrixr>& K, cv::vectorr& k) {
    ASSERT(A.rows() == 3 && A.cols() == 3);
    ASSERT(k.length() == 2);
    ASSERT(!K.empty());
    for (auto const& k_ : K) {
        ASSERT(k_.rows() == 3 && k_.cols() == 4);
    }
    std::vector<real_t> x(7 + 12 * K.size());
    auto xIter = x.begin();
    *xIter++ = A(0, 0);  // a
    *xIter++ = A(1, 1);  // b
    *xIter++ = A(0, 1);  // c
    *xIter++ = A(0, 2);  // u0
    *xIter++ = A(1, 2);  // v0
    *xIter++ = k[0];
    *xIter++ = k[1];

    for (auto const& k_ : K) {
        for (int i = 0; i < 3; ++i) {
            *xIter++ = k_(i, 0);  // r11, r21, r31
            *xIter++ = k_(i, 1);  // r12, r22, r32
            *xIter++ = k_(i, 2);  // r13, r23, r33
            *xIter++ = k_(i, 3);  // t1,  t2,  t3
        }
    }


    return x;
}

void unwindX(real_t* x, cv::matrixr& A, std::vector<cv::matrixr>& K, cv::vectorr& k) {
    A(0, 0) = *x++;  // a
    A(1, 1) = *x++;  // b
    A(0, 1) = *x++;  // c
    A(0, 2) = *x++;  // u0
    A(1, 2) = *x++;  // v0
    k[0] = *x++;
    k[1] = *x++;

    for (auto& k_ : K) {
        for (int i = 0; i < 3; ++i) {
            k_(i, 0) = *x++;  // r11, r21, r31
            k_(i, 1) = *x++;  // r12, r22, r32
            k_(i, 2) = *x++;  // r13, r23, r33
            k_(i, 3) = *x++;  // t1,  t2,  t3
        }
    }
}

void reconstruct_Ak_from_X(const real_t* x,
                           real_t& a,
                           real_t& b,
                           real_t& c,
                           real_t& u0,
                           real_t& v0,
                           real_t& k1,
                           real_t& k2) {
    a = x[0];
    b = x[1];
    c = x[2];
    u0 = x[3];
    v0 = x[4];
    k1 = x[5];
    k2 = x[6];
}

void reconstruct_K_from_X(const real_t* x,
                          int board_id,
                          real_t& r11,
                          real_t& r12,
                          real_t& r13,
                          real_t& r21,
                          real_t& r22,
                          real_t& r23,
                          real_t& r31,
                          real_t& r32,
                          real_t& r33,
                          real_t& t1,
                          real_t& t2,
                          real_t& t3) {
    const real_t* x_K = x + 7 + 12 * board_id;
    r11 = *x_K++;
    r12 = *x_K++;
    r13 = *x_K++;
    t1 = *x_K++;
    r21 = *x_K++;
    r22 = *x_K++;
    r23 = *x_K++;
    t2 = *x_K++;
    r31 = *x_K++;
    r32 = *x_K++;
    r33 = *x_K++;
    t3 = *x_K++;
}

static
int optimize_fcn(void* p, int m, int n, const real_t* x, real_t* fvec, real_t* fjac, int ldfjac, int iflag) {
    assert(m % 2 == 0);
    if (iflag == 0) {
        return 0;
    }

    const auto data = (CalibrationOptimizationData*)(p);
    const auto target_pts = data->target_pts;
    const auto target_pts_nrm = data->target_pts_nrm;
    const auto model_pts = data->model_pts;
	const auto board_count = data->board_count;
    const auto point_count = m / 2;
	const auto model_point_count = model_pts.size();

    real_t a, b, c, u0, v0, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3, k1, k2;

    reconstruct_Ak_from_X(x, a, b, c, u0, v0, k1, k2);

    if (iflag == 1) {
        // calculate residual
		int ptId = 0;
        for (int i = 0; i < board_count; ++i) {
			const auto& t_n = target_pts_nrm[i];
			const auto& t = target_pts[i];
			reconstruct_K_from_X(x, i, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3);
			for (int j = 0; j < model_point_count; ++j, ++ptId) {
				const auto x = model_pts[j][0];
				const auto y = model_pts[j][1];
				const auto z = model_pts[j][2];

				const auto xn = t_n[j][0];
				const auto yn = t_n[j][1];
				const auto x_yn = xn*xn + yn*yn;

				const auto m1_K = r11*x + r12*y + r13*z + t1;
				const auto m2_K = r21*x + r22*y + r23*z + t2;
				const auto m3_K = r31*x + r32*y + r33*z + t3;

				const auto m1_A = (a*m1_K + c*m2_K + u0*m3_K) / m3_K;
				const auto m2_A = (b*m2_K + v0*m3_K) / m3_K;

				const auto z_ = (k1*x_yn + k2*(x_yn*x_yn));
				const auto u = m1_A + (m1_A - u0)*z_;
				const auto v = m2_A + (m2_A - v0)*z_;

				const auto dx = u - t[j][0];
				const auto dy = v - t[j][0];

				fvec[ptId * 2 + 0] = dx*dx;
				fvec[ptId * 2 + 1] = dy*dy;
			}
        }
    } else if (iflag == 2) {
        // calculate jacobian
		int ptId = 0;
		for (int i = 0; i < board_count; ++i) {
			const auto& t_n = target_pts_nrm[i];
			const auto& t = target_pts[i];
			reconstruct_K_from_X(x, i, r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3);
			for (int j = 0; j < model_point_count; ++j, ++ptId) {
				const auto x = model_pts[j][0];
				const auto y = model_pts[j][1];
				const auto z = model_pts[j][2];

				const auto xn = t_n[j][0];
				const auto yn = t_n[j][1];
				const auto x_yn = xn*xn + yn*yn;

				const auto m1_K = r11*x + r12*y + r13*z + t1;
				const auto m2_K = r21*x + r22*y + r23*z + t2;
				const auto m3_K = r31*x + r32*y + r33*z + t3;

				const auto m1_A = (a*m1_K + c*m2_K + u0*m3_K) / m3_K;
				const auto m2_A = (b*m2_K + v0*m3_K) / m3_K;

				const auto z_ = (k1*x_yn + k2*(x_yn*x_yn));

				real_t m1_dA[3];
				real_t m1_dk[2];
				real_t m1_dK[12];

				real_t m2_dA[3];
				real_t m2_dk[2];
				real_t m2_dK[12];

				m1_dA[0] = m1_K / m3_K; // a
				m1_dA[1] = 0.0; // b
				m1_dA[2] = m2_K / m3_K; // c
				real_t m1_du0 = 1.0; // u0
				real_t m1_dv0 = 0.0; // v0

				m1_dk[0] = 0.0; // k1
				m1_dk[1] = 0.0; // k2

				real_t m1_t1 = a / m3_K; // t1
				m1_dK[0] = x*m1_t1; // r11
				m1_dK[1] = y*m1_t1; // r12
				m1_dK[2] = z*m1_t1; // r13
				m1_dK[3] = m1_t1; // t1
				real_t m1_t2 = c / m3_K;
				m1_dK[4] = x*m1_t2; // r21
				m1_dK[5] = y*m1_t2; // r22
				m1_dK[6] = z*m1_t2; // r23
				m1_dK[7] = m1_t2; // t2
				real_t m1_t3 = -(a*m1_K + c*m2_K) / (m3_K*m3_K);
				m1_dK[8] = x*m1_t3; // r31
				m1_dK[9] = y*m1_t3; // r32
				m1_dK[10] = z*m1_t3; // r33
				m1_dK[11] = m1_t3; // t3

				m2_dA[0] = 0.0; // a
				m2_dA[1] = m2_K / m3_K; // b
				m2_dA[2] = 0.0; // c
				real_t m2_du0 = 0.0; // u0
				real_t m2_dv0 = 1.0; // v0

				m2_dk[0] = 0.0; // k1
				m2_dk[1] = 0.0; // k2

				m2_dK[0] = 0.0; // r11
				m2_dK[1] = 0.0; // r12
				m2_dK[2] = 0.0; // r13
				m2_dK[3] = 0.0; // t1
				real_t m2_t2 = b / m3_K;
				m2_dK[4] = x*m2_t2; // r21
				m2_dK[5] = y*m2_t2; // r22
				m2_dK[6] = z*m2_t2; // r23
				m2_dK[7] = m2_t2; // t1
				real_t m2_t3 = -(b*m2_K) / (m3_K*m3_K);
				m2_dK[8] = x*m2_t3; // r31
				m2_dK[9] = y*m2_t3; // r32
				m2_dK[10] = z*m2_t3; // r33
				m2_dK[11] = m2_t3; // t3

				real_t z_dK[2];
				z_dK[0] = xn*xn + yn*yn;
				z_dK[1] = z_dK[0] * z_dK[0];

				real_t u_dA[3];
				real_t v_dA[3];

				real_t u_dk[2];
				real_t v_dk[2];

				real_t u_dK[12];
				real_t v_dK[12];

				for (int k = 0; k < 3; ++k) {
					u_dA[k] = m1_dA[k] * (1.0 + z_);
					v_dA[k] = m2_dA[k] * (1.0 + z_);
				}

				real_t u_du0 = m1_du0 + z_*(m1_du0 - 1.0);
				real_t u_dv0 = m1_dv0 + z_*m1_dv0;
				real_t v_du0 = m2_du0 + z_*m2_du0;
				real_t v_dv0 = m2_dv0 + z_*(m2_dv0 - 1.0);

				for (int k = 0; k < 2; ++k) {
					u_dk[k] = (m1_A - u0)*z_dK[k];
					v_dk[k] = (m2_A - v0)*z_dK[k];
				}

				for (int k = 0; k < 12; ++k) {
					u_dK[k] = m1_dK[k] * (1.0 + z_);
					v_dK[k] = m2_dK[k] * (1.0 + z_);
				}

				auto jac_u = fjac + ptId * 2 * n;
				auto jac_v = fjac + (ptId * 2 + 1)*n;

				// copy u_dA
				jac_u[0] = u_dA[0];
				jac_u[1] = u_dA[1];
				jac_u[2] = u_dA[2];
				jac_u[3] = u_du0;
				jac_u[4] = u_dv0;
				jac_u[5] = u_dk[0];
				jac_u[6] = u_dk[1];

				jac_v[0] = v_dA[0];
				jac_v[1] = v_dA[1];
				jac_v[2] = v_dA[2];
				jac_v[3] = v_du0;
				jac_v[4] = v_dv0;
				jac_v[5] = v_dk[0];
				jac_v[6] = v_dk[1];

				jac_u += 7; // jump to the beginning of the K
				for (int k = 0; k < board_count; ++k, jac_u += 12, jac_v += 12) {
					if (k == i) {
						std::memcpy(jac_u, u_dK, 12 * sizeof(real_t));
						std::memcpy(jac_v, v_dK, 12 * sizeof(real_t));
					} else {
						std::memset(jac_u, 0, 12 * sizeof(real_t));
						std::memset(jac_v, 0, 12 * sizeof(real_t));
					}
				}
			}
		}
    }

    return 0;
}

int optimize_calib(const std::vector<std::vector<cv::vec2r> >& image_points,
                   const std::vector<std::vector<cv::vec2r> >& image_points_nrm,
                   const std::vector<cv::vec3r>& model_points,
                   cv::matrixr& A,
                   std::vector<cv::matrixr>& K,
                   cv::vectorr& k,
                   bool fixed_aspect,
                   bool no_skew,
                   real_t tol) {
    ASSERT(image_points.front().size() == model_points.size());
    ASSERT(image_points.size() == K.size());

    const auto point_count = model_points.size();
    const auto board_count = K.size();

    CalibrationOptimizationData data {image_points,
                                      image_points_nrm,
                                      model_points,
                                      board_count};

    std::vector<real_t> x = setupX(A, K, k);

    int m = point_count * 2;
    int n = x.size();

    int info = 0;
    int nfev, njev;

    std::vector<int> ipvt(n);
    std::vector<real_t> wa1(n), wa2(n), wa3(n), wa4(m);
    std::vector<real_t> diag(n);
    std::vector<real_t> fvec(m), fjac(m * n);
    std::vector<real_t> qtf(n);

    int maxfev = 100;

    info = lmder(
        optimize_fcn, &data, m, n, x.data(),
        fvec.data(), fjac.data(), m,
        1e-8, 1e-12, 0.0, maxfev, diag.data(),
        1, 100.0, 1, &nfev, &njev, ipvt.data(), qtf.data(),
        wa1.data(), wa2.data(), wa3.data(), wa4.data());

    return info;
}
