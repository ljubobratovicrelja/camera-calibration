#include "utils.h"


std::vector<std::string> split_string(std::string s, const std::string& delimiter) {
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

cv::image_array undistort_image(const cv::image_array& image, const cv::matrixr& A, const cv::vectorr& k) {
    cv::image_array undist = cv::matrix3b::zeros(image.rows(), image.cols());

    auto fx = A(0, 0);
    auto fy = A(1, 1);
    auto cx = A(0, 2);
    auto cy = A(1, 2);

    #pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < image.rows(); ++i) {
        for (unsigned j = 0; j < image.cols(); ++j) {
            cv::vec3r proj_ptn = {(static_cast<real_t>(j) - cx) / fx, (static_cast<real_t>(i) - cy) / fy, 1.};

            if (k.length() == 4) {
                real_t r2 = proj_ptn[0] * proj_ptn[0] + proj_ptn[1] * proj_ptn[1] + 1;
                real_t d_r = (1 + k[0] * r2 + k[1] * (r2 * r2));  // radial distortion
                real_t d_t = 2 * k[2] * proj_ptn[0] * proj_ptn[1] + k[3] * (r2 + 2 * (proj_ptn[0] * proj_ptn[0]));  // tan
                                                                                                                    // distortion
                proj_ptn[0] = proj_ptn[0] * d_r + d_t;
                proj_ptn[1] = proj_ptn[1] * d_r + d_t;
            } else if (k.length() == 8) {
                real_t r2 = proj_ptn[0] * proj_ptn[0] + proj_ptn[1] * proj_ptn[1] + 1;
                real_t r3 = proj_ptn[0] * proj_ptn[0] * proj_ptn[0] + proj_ptn[1] * proj_ptn[1] * proj_ptn[1] + 1;
                real_t k_u = 1 + k[0] * r2 + k[1] * (r2 * r2) + k[2] * (r3 * r3);
                real_t k_d = 1 + k[3] * r2 + k[4] * (r2 * r2) + k[5] * (r3 * r3);
                real_t d_r = (k_d) ? k_u / k_d : 0.;  // radial distortion
                real_t d_t = 2 * k[2] * proj_ptn[0] * proj_ptn[1] + k[3] * (r2 + 2 * (proj_ptn[0] * proj_ptn[0]));  // tan
                                                                                                                    // distortion
                proj_ptn[0] = proj_ptn[0] * d_r + d_t;
                proj_ptn[1] = proj_ptn[1] * d_r + d_t;
            }

            auto x_undist = proj_ptn[0] * fx + cx;
            auto y_undist = proj_ptn[1] * fy + cy;

            if ((x_undist < 0) || (x_undist >= image.cols()) ||
                (y_undist < 0) || (y_undist >= image.rows())) {
                continue;
            }

            undist.at<byte>(i, j, 0) = image.at<byte>(y_undist, x_undist, 0);
            undist.at<byte>(i, j, 1) = image.at<byte>(y_undist, x_undist, 1);
            undist.at<byte>(i, j, 2) = image.at<byte>(y_undist, x_undist, 2);
        }
    }

    return undist;
}

void string_to_double(double& val, int argc, char** argv, int i, const std::string& val_name) {
    if (i + 1 >= argc) {
        std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
        exit(EXIT_FAILURE);
    }
    try {
        val = std::stod(argv[i + 1]);
    } catch (std::invalid_argument& e)   {
        std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::out_of_range& e)   {
        std::cout << "Argument error: " << val_name << " cannot be converted to double value - out of range.\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void string_to_int(int& val, int argc, char** argv, int i, const std::string& val_name) {
    if (i + 1 >= argc) {
        std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
        exit(EXIT_FAILURE);
    }
    try {
        val = std::stoi(argv[i + 1]);
    } catch (std::invalid_argument& e)   {
        std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
        exit(EXIT_FAILURE);
    } catch (std::out_of_range& e)   {
        std::cout << "Argument error: " << val_name << " cannot be converted to double value - out of range.\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}
