#include "homography.hpp"

#include <cassert>
#include <linalg.hpp>
#include <math.hpp>

#include <cminpack.h>


cv::matrixr homography_solve(const std::vector<cv::vec2r>& image_points, const std::vector<cv::vec3r>& model_points) {
    ASSERT(image_points.size() >= 4 && image_points.size() == model_points.size());

    cv::matrixr A = cv::matrixr::zeros(2 * image_points.size(), 9);

    for (unsigned i = 0; i < image_points.size(); ++i) {
        A(i * 2 + 0, 0) = -model_points[i][0];
        A(i * 2 + 0, 1) = -model_points[i][1];
        A(i * 2 + 0, 2) = -1;
        A(i * 2 + 0, 6) = image_points[i][0] * model_points[i][0];
        A(i * 2 + 0, 7) = image_points[i][0] * model_points[i][1];
        A(i * 2 + 0, 8) = image_points[i][0];

        A(i * 2 + 1, 3) = -model_points[i][0];
        A(i * 2 + 1, 4) = -model_points[i][1];
        A(i * 2 + 1, 5) = -1;
        A(i * 2 + 1, 6) = image_points[i][1] * model_points[i][0];
        A(i * 2 + 1, 7) = image_points[i][1] * model_points[i][1];
        A(i * 2 + 1, 8) = image_points[i][1];
    }

    cv::matrixr H;
    cv::null_solve(A, H);

    H.reshape(3, 3);

    return H / H(2, 2);
}

struct HomographyOptimizationData {
    const cv::vec2r* target_pts;
    const cv::vec3r* model_pts;
};


static
int optimize_fcn(void* p, int m, int n, const real_t* x, real_t* fvec, real_t* fjac, int ldfjac, int iflag) {
    assert(m % 2 == 0);
    if (iflag == 0) {
        return 0;
    }

    auto data = (HomographyOptimizationData*)(p);
    auto target_pts = data->target_pts;
    auto model_pts = data->model_pts;

    const auto point_count = m / 2;

    real_t h11 = x[0];
    real_t h12 = x[1];
    real_t h13 = x[2];
    real_t h21 = x[3];
    real_t h22 = x[4];
    real_t h23 = x[5];
    real_t h31 = x[6];
    real_t h32 = x[7];
    real_t h33 = x[8];

    if (iflag == 1) {
        // calculate residual
        for (int i = 0; i < point_count; ++i) {
            const auto& t = target_pts[i];
            const auto& m = model_pts[i];
            auto p_x = h11 * m[0] + h12 * m[1] + h13;
            auto p_y = h21 * m[0] + h22 * m[1] + h23;
            auto p_w = h31 * m[0] + h32 * m[1] + h33;
            p_x /= p_w;
            p_y /= p_w;
            const auto d_x = t[0] - p_x;
            const auto d_y = t[1] - p_y;
            fvec[i * 2] = d_x * d_x;
            fvec[i * 2 + 1] = d_y * d_y;
        }
    } else if (iflag == 2) {
        // calculate jacobian
        for (int i = 0; i < point_count; ++i) {
            real_t x = target_pts[i][0], y = target_pts[i][1];
            real_t v1 = h11 * x + h12 * y + h13;
            real_t v2 = h21 * x + h22 * y + h23;
            real_t v3 = h31 * x + h32 * y + h33;
            real_t v3_2 = v3 * v3;

            real_t* jac = fjac + (i * 2 * n);
            jac[0] = x / v3;
            jac[1] = y / v3;
            jac[2] = 1.0 / v3;
            jac[3] = 0;
            jac[4] = 0;
            jac[5] = 0;
            jac[6] = -(x * v1) / v3_2;
            jac[7] = -(y * v1) / v3_2;
            jac[8] = -v1 / v3_2;

            jac += n;
            jac[0] = 0;
            jac[1] = 0;
            jac[2] = 0;
            jac[3] = x / v3;
            jac[4] = y / v3;
            jac[5] = 1.0 / v3;
            jac[6] = -(x * v2) / v3_2;
            jac[7] = -(y * v2) / v3_2;
            jac[8] = -v2 / v3_2;
        }
    }

    return 0;
}

int homography_optimize(const std::vector<cv::vec2r>& image_points,
                        const std::vector<cv::vec3r>& model_points,
                        cv::matrixr& H,
                        real_t tol) {
    ASSERT(image_points.size() == model_points.size());

    const auto point_count = image_points.size();

    HomographyOptimizationData data;
    data.model_pts = model_points.data();
    data.target_pts = image_points.data();

    int m = point_count * 2;
    int n = 9;

    int info = 0;
    int nfev, njev;

    std::vector<int> ipvt(n);
    std::vector<real_t> wa1(n), wa2(n), wa3(n), wa4(m);
    std::vector<real_t> diag(n);
    std::vector<real_t> fvec(m), fjac(m * n);
    std::vector<real_t> qtf(n);

    int maxfev = 1000;

    info = lmder(
        optimize_fcn, &data, m, n, H.data(),
        fvec.data(), fjac.data(), m,
        1e-8, 1e-12, 0.0, maxfev, diag.data(),
        1, 100.0, 1, &nfev, &njev, ipvt.data(), qtf.data(),
        wa1.data(), wa2.data(), wa3.data(), wa4.data());

    H /= H(2, 2);

    return info;
}
