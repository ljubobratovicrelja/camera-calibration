#include <io.hpp>
#include <gui.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>

#include "homography.hpp"
#include "optimize.hpp"
#include "calib.hpp"
#include "routines.h"
#include "calibio.h"
#include "utils.h"
#include "argparser.h"



void detection_routine(double ang_thresh, double mag_thresh, int nn_count, int p_rows, int p_cols, std::string im_files_path);


int main(int argc, char **argv) {

    std::string pattern_file = "";
    bool fixed_aspect = false;
    bool no_skew = false;
    bool skip_optimization = false;
    bool skip_extrinsic_optmization = false;
    bool skip_distortion_optimization = false;
    double ftol = 1e-05;
    double model_square_size = 3.; // 3cm is default size of the chessboard calibration pattern printed on A4
    int p_rows = 6;
    int p_cols = 9;
    bool no_distortion = false;
	bool detection = false;
	double ang_thresh = 5.;
	double mag_thresh = 0.15;
	int nn_count = 8;
	std::string im_files_path;
	std::vector<std::string> im_files;

	// define argument parser
	arg_parser argp;

	argp
		.def_argument({ detection, "detection", "dt", "Pattern detection program." })
		.def_argument({ ang_thresh, "ang-thresh", "at", "Angular threshold used to determine bad connection in pattern search." })
		.def_argument({ mag_thresh, "mag-thresh", "mt", "Magnitude threshold used to determine bad connection in pattern search." })
		.def_argument({ nn_count, "nn-count", "nc", "Nearest neighbour count used in pattern search." })
		.def_argument({ im_files_path, "im-files", "imf", "Path to a file containing list of pattern projection (images)." })
		.def_argument({ p_rows, "p-rows", "pr", "Number of rows in the pattern." })
		.def_argument({ p_cols, "p-cols", "pc", "Number of columns in the pattern." })
		.def_argument({ no_skew, "no-skew", "sk", "Ignore skew parameter estimation during calibration procedure." })
		.def_argument({ fixed_aspect, "fixed-aspect", "fa", "Consider aspect ratio to be fixed to 1:1." })
		.def_argument({ ftol, "ftol", "ft", "Optimization convergence tolerance value." })
		.def_argument({ model_square_size, "model-square-size", "msq", "Size of the pattern square size in cm." })
		.def_argument({ skip_optimization, "skip-opt", "sko", "Skip global optimization (use only analytical solution)." })
		.def_argument({ skip_extrinsic_optmization, "skip-ext-opt", "ske", "Skip optimization of extrinsic parameters." })
		.def_argument({ skip_distortion_optimization, "skip-dist-opt", "skd", "Skip optimization of distortion parameters." })
		.def_argument({ no_distortion, "no-dist", "nod", "Don't estimate distortion parameters." })
		.def_argument({ pattern_file, "pattern-file", "pf", "Path to the file containing pattern projections." });

    if (argc < 2) {
		argp.print_help("Invalid arguments.");
        exit(0);
    }

    std::cout << "********************************************" << std::endl;

    std::cout << "Program ran using flags:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << "\n********************************************" << std::endl << std::endl;

    // ================= PARSE ARGUMENTS, INITIALIZE PROGRAM ================== //

	try {
		argp.parse(argc, argv);
	} catch (std::exception &e) {
		argp.print_help(std::string() + "Parsing arguments failed: " + e.what());
		exit(1);
	}

    // ================= DETECTION ROUTINE, INITIALIZE PROGRAM ================== //

    if (detection) {
		detection_routine(ang_thresh, mag_thresh, nn_count, p_rows, p_cols, im_files_path);
		exit(0);
    }

	// =================== INITIALIZE CALIBRATION ============================= //

    unsigned im_w, im_h; // image width and height

    std::vector<std::vector<cv::vec2r>> image_points_orig, image_points_nrm;
    std::vector<cv::vec3r> model_points;

	// Read pattern data
	if (!read_custom_data(pattern_file, image_points_orig, model_points, im_w, im_h, model_square_size)) {
		std::cout << "Error while reading pattern file at: " << pattern_file << std::endl;
		return EXIT_FAILURE;
	}

    auto image_points_count = image_points_orig.size();

	// normalize image points
    image_points_nrm = image_points_orig; 
    auto N = normalize_image_points(image_points_nrm, im_w, im_h);
    auto N_inv = N.clone();
    cv::invert(N_inv);

	// Calculate homographies
    std::vector<cv::matrixr> Hs(image_points_count);
    for (unsigned i = 0; i < image_points_count; ++i) {
        ASSERT(image_points_nrm[i].size() == model_points.size());
        Hs[i] = homography_solve(image_points_nrm[i], model_points);
        homography_optimize(image_points_nrm[i], model_points, Hs[i], ftol);
    }

	// Calculate intrinsics, based on homographies between image projections and model.
    auto A_p = compute_intrisics(Hs); 

    if (!A_p) {
        std::cout << "Failure calculating intrinsic parameters." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Intrinsics matrix A':" << std::endl;
    std::cout << A_p << std::endl;

	// Denormalize intrinsic parameters, to achieve real-scale values.
    auto A = denormalize_intrinsics(A_p, N);

	// Regulate intrinsics if fixed aspect ratio is chosen, or ignoring of skew parameter.
    if (fixed_aspect) {
        auto asp = (A(0, 0) + A(1, 1)) / 2.;
        A(0, 0) = A(1, 1) = asp;
    }

    if (no_skew) {
        A(0, 1) = 0.0;
    }

    std::cout << "Denormalized intrinsics matrix A:" << std::endl;
    std::cout << A << std::endl;

	// Calculate extrinsics

    std::vector<std::vector<cv::vec2r>> image_points_proj(image_points_count);
    std::vector<std::vector<cv::vec3r>> camera_points(image_points_count);

    std::vector<cv::matrixr> Ks;

    for (unsigned i = 0; i < image_points_count; ++i) {
        auto K = compute_extrinsics(A, N_inv*Hs[i]);
        auto err = calc_reprojection(A, K, model_points, image_points_orig[i], image_points_proj[i]);
        std::cout << "Extrinsics " <<  i << std::endl;
        std::cout << "\nReprojection error: " << err << std::endl;
        std::cout << "K:\n" << K << std::endl;

        Ks.push_back(K);
    }

	// Calculate distrotion

    cv::vectorr k;

    if (!skip_optimization) 
        optimize_calib(image_points_orig, model_points, A, Ks, k, fixed_aspect, no_skew, ftol);

    if (!no_distortion) {
        k = compute_distortion(image_points_orig, image_points_nrm, image_points_proj, A)(0, 1);
        std::cout << "k:\n" << k << std::endl << std::endl;

        if (!skip_distortion_optimization) {
            optimize_calib(image_points_orig, model_points, A, Ks, k, fixed_aspect, no_skew, ftol);
        }
    } else {
        std::cout << "Distortion skipped" << std::endl;
    }

    std::cout << "\n\n**********************************************************" << std::endl;
    std::cout << "Final Optimization Results:" << std::endl;
    std::cout << "A:\n" << A << std::endl;
    std::cout << "k:\n" << k << std::endl << std::endl;

    real_t mean_err = 0.;

    for (unsigned i = 0; i < image_points_count; ++i) {
        std::cout << "------------ K no." << i << " --------------\n" << Ks[i] << std::endl;
        auto err = calc_reprojection(A, Ks[i], model_points, image_points_orig[i], image_points_proj[i], k);
        mean_err += err;
        std::cout << "Reprojection error: " << err << std::endl << std::endl;

        real_t scale = (im_w > 1000) ? 1000. / im_w : 1.;
        auto reproj = draw_reprojection(image_points_orig[i], image_points_proj[i], im_w, im_h, scale);

        cv::imwrite(reproj, "reprojection_" + std::to_string(i) + ".png");

#ifndef CV_IGNORE_GUI
        cv::imshow("reprojection",reproj);
        cv::wait_key();
#endif
    }

    /*
    auto img = cv::imread("/home/relja/CalibIm3.jpg");

    if (img) {
        
        auto undist = undistort_image(img, A, k);
        cv::imwrite(undist, "/home/relja/CalibIm3_undistort.jpg");
    }
    */

    write_results(A, k, Ks);

    std::cout << "Mean reprojection error for all patterns: " << (mean_err/image_points_count) << std::endl;

    std::cout << "**********************************************************" << std::endl;

    return EXIT_SUCCESS;
}


void detection_routine
(
	double ang_thresh,
	double mag_thresh,
	int nn_count,
	int p_rows,
	int p_cols,
	std::string im_files_path
) {
	// cap values
	if (ang_thresh <= 0) {
		std::cout << "Invalid value for ang-thresh - should be non-negative, non-zero." << std::endl;
		ang_thresh = 15.;
	}
	if (mag_thresh <= 0) {
		std::cout << "Invalid value for mag-thresh - should be non-negative, non-zero." << std::endl;
		mag_thresh = 0.15;
	}
	if (nn_count <= 0) {
		std::cout << "Invalid value for nn-count - should be non-negative, non-zero." << std::endl;
		nn_count = 10;
	}
	if (p_rows <= 2) {
		std::cout << "Pattern rows must be > 2" << std::endl;
		p_rows = 6;
	}
	if (p_cols <= 2) {
		std::cout << "Pattern cols must be > 2" << std::endl;
		p_cols = 6;
	}

	auto im_files = read_image_collection(im_files_path);

	if (im_files.empty()) {
		std::cout << "Image files not read properly." << std::endl;
		exit(-1);
	}
	else {
		std::cout << "Read " << im_files.size() << " paths to image files from " << im_files_path << " file." << std::endl;
	}

	std::vector<cv::matrixr> pattern_ims;

	if (im_files.empty()) {
		std::cout << "=========================================" << std::endl;
		std::cout << "Enter image files with captured patterns," << std::endl;
		std::cout << "* For end write \"end\"\n* For clear last entry type \"cl\"" << std::endl;
		std::cout << "=========================================" << std::endl;

		std::string in;
		std::string path;

		std::cout << "1. Enter root folder for images:" << std::endl;
		std::cin >> path;

		std::cout << "2. Enter image filenames:" << std::endl;

		while (true) {
			std::cin >> in;
			if (in == "end") {
				break;
			}
			else if (in == "cl") {
				if (pattern_ims.size()) {
					pattern_ims.pop_back();
				}
			}

			in = path + "/" + in;
			cv::matrixr im;

			try {
				im = cv::imread(in, cv::REAL, 1);
			}
			catch (std::runtime_error &e) {
				std::cout << "Unexpected error occurred while reading image:\n" << in << std::endl;
				std::cout << "Error message: " << e.what() << std::endl;
				continue;
			}

			if (!im) {
				std::cout << "Failure loading image at: \n" << in << std::endl;
			}
			else {
				if (im.cols() > 1500) {
					real_t scale_factor = 1500. / im.cols();
					cv::resize(im, im, im.rows()*scale_factor, im.cols()*scale_factor);
				}
				if (!pattern_ims.empty()) {
					// check pattern size
					if (im.size() != pattern_ims.back().size()) {
						std::cout << "Error - pattern not of the same size as previous - pattern rejected!" << std::endl;
						continue;
					}
				}
				pattern_ims.push_back(im);
				std::cout << "Image at " << in << " accepted." << std::endl;
			}
		}
	}
	else {
		for (int i = 0; i < im_files.size(); ++i) {
			cv::matrixr im;
			std::string &in = im_files[i];
			try {
				im = cv::imread(in, cv::REAL, 1);
			}
			catch (std::runtime_error &e) {
				std::cout << "Unexpected error occurred while reading image:\n" << in << std::endl;
				std::cout << "Error message: " << e.what() << std::endl;
				continue;
			}
			if (!im) {
				std::cout << "Failure loading image at: \n" << in << std::endl;
			}
			else {
				if (im.cols() > 800) {
					real_t scale_factor = 800. / im.cols();
					cv::resize(im, im, im.rows()*scale_factor, im.cols()*scale_factor);
				}
				if (!pattern_ims.empty()) {
					// check pattern size
					if (im.size() != pattern_ims.back().size()) {
						std::cout << "Error - pattern not of the same size as previous - pattern rejected!" << std::endl;
						continue;
					}
				}

				pattern_ims.push_back(im);
				std::cout << "Image at " << in << " accepted." << std::endl;
			}
		}
	}

	if (pattern_ims.empty()) {
		std::cout << "Pattern data empty - detection exiting..." << std::endl;
		return;
	}

	std::string out;
	std::cout << "Where to write detected pattern point data?" << std::endl;
	std::cin >> out;

	std::cout << "=========================================" << std::endl;
	std::cout << "Running pattern detection..." << std::endl;

	pattern_detection(pattern_ims, out, p_rows, p_cols, ang_thresh, mag_thresh, nn_count);

	std::cout << "Pattern detection finished successfully..." << std::endl;
	std::cout << "=========================================" << std::endl;

}
