#include <io.hpp>
#include <gui.hpp>

#include <iostream>
#include <fstream>
#include <cstring>

#include "calibpattern.hpp"
#include "homography.hpp"
#include "optimize.hpp"
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

void pattern_detection(const std::vector<cv::matrixr> &pattern_ims, const std::string &out_path, unsigned p_rows, unsigned p_cols, real_t ang_thresh, real_t mag_thresh, unsigned nn_count) {

	std::vector<std::vector<cv::vec2r> > patterns;

	unsigned im_w = 0, im_h = 0;

	for (auto image: pattern_ims) {

		im_w = image.cols();
		im_h = image.rows();

		cv::matrixr im_r = image, im_rb;
		cv::matrixr gauss_k = cv::gauss({3, 3}, 3);

		im_rb = cv::conv(im_r, gauss_k);

		auto pattern = detect_pattern(im_rb, p_rows, p_cols, ang_thresh, mag_thresh, nn_count);

		if (pattern.size() == p_rows*p_cols) {
			pattern = sub_pixel_detect(pattern, im_r, {5, 5});

			cv::matrix3r im_draw = im_r.clone();
			draw_chessboard(im_draw, pattern, cv::vec3r(255.f, 0.f, 0.f));

#ifndef CV_IGNORE_GUI
			std::cout << "Accept pattern? (Y/n)" << std::endl;

			cv::imshow("pattern", im_draw);
			auto r = cv::wait_key();

			if (r == 'n') {
				std::cout << "Pattern rejected." << std::endl;
				continue;
			} else {
				std::cout << "Pattern accepted." << std::endl;
				patterns.push_back(pattern);
			}
#else
			std::cout << "Pattern found." << std::endl;
			patterns.push_back(pattern);
#endif
		} else {
			std::cout << "Pattern not found" << std::endl;
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

int main(int argc, char **argv) {

	std::cout << "********************************************" << std::endl;

	std::cout << "Program ran using flags:" << std::endl;
	for (int i = 0; i < argc; ++i) {
		std::cout << argv[i] << " ";
	}
	std::cout << "\n********************************************" << std::endl << std::endl;

	std::string pattern_file = "";
	bool fixed_aspect = false;
	bool no_skew = false;
	bool skip_optimization = false;
	bool skip_extrinsic_optmization = false;
	bool skip_distortion_optimization = false;
	double ftol = 1e-14;
	double model_square_size = 3.; // 3cm is default size of the chessboard calibration pattern printed on A4
	int p_rows = 6;
	int p_cols = 9;

	// ================= PARSE ARGUMENTS, INITIALIZE PROGRAM ================== //

	if (argc < 2) {
		std::cout << "Invalid arguments:\n" << std::endl;
		std::cout << "Flags:\n--detection: pattern detection program;\n"
		          << "--zhang: calibrate using zhang experimental data;\n"
		          << "path to pattern file generated by --detection program."	<< std::endl;
	}

	// Read string to double with error proofing.
	auto string_to_double = [](double &val, int argc, char **argv, int i, const std::string &val_name) {
		if (i + 1 >= argc) {
			std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
			exit(EXIT_FAILURE);
		}
		try {
			val = std::stod(argv[i + 1]);
		} catch (std::invalid_argument &e) {
			std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
			exit(EXIT_FAILURE);
		} catch (std::out_of_range &e) {
			std::cout << "Argument error: " << val_name << " cannot be converted to double value - out of range.\n" << std::endl;
			exit(EXIT_FAILURE);
		}
	};

	// Read string to int with error proofing.
	auto string_to_int = [](int &val, int argc, char **argv, int i, const std::string &val_name) {
		if (i + 1 >= argc) {
			std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
			exit(EXIT_FAILURE);
		}
		try {
			val = std::stoi(argv[i + 1]);
		} catch (std::invalid_argument &e) {
			std::cout << "Argument error: " << val_name << " flag needs to have float value following." << std::endl;
			exit(EXIT_FAILURE);
		} catch (std::out_of_range &e) {
			std::cout << "Argument error: " << val_name << " cannot be converted to double value - out of range.\n" << std::endl;
			exit(EXIT_FAILURE);
		}
	};

	if (argv[1] == std::string("--detection")) {

		double ang_thresh = 5.;
		double mag_thresh = 0.15;
		int nn_count = 8;
		std::vector<std::string> im_files;

		if (argc > 2) {
			for (int i = 2; i < argc; ++i) {
				if (argv[i] == std::string("--ang-thresh")) {
					string_to_double(ang_thresh, argc, argv, i, "ang-thresh");
					if (ang_thresh <= 0) {
						std::cout << "Invalid value for ang-thresh - should be non-negative, non-zero." << std::endl;
						ang_thresh = 15.;
					}
				} else if (argv[i] == std::string("--mag-thresh")) {
					string_to_double(mag_thresh, argc, argv, i, "mag-thresh");
					if (mag_thresh <= 0) {
						std::cout << "Invalid value for mag-thresh - should be non-negative, non-zero." << std::endl;
						mag_thresh = 0.15;
					}
				} else if (argv[i] == std::string("--nn-count")) {
					string_to_int(nn_count, argc, argv, i, "nn-count");
					if (nn_count <= 0) {
						std::cout << "Invalid value for nn-count - should be non-negative, non-zero." << std::endl;
						nn_count = 10;
					}
				} else if (argv[i] == std::string("--im-files")) {
					if (i + 1 >= argc)
						std::cout << "Invalid image file collection argument - should have path to the file." << std::endl;

					im_files = read_image_collection(argv[i + 1]);

					if (im_files.empty()) {
						std::cout << "Image files not read properly." << std::endl;
					} else {
						std::cout << "Read " << im_files.size() << " paths to image files from "
						          << argv[i + 1] << " file." << std::endl;
					}
				} else if (argv[i] == std::string ("--p-rows")) {
					string_to_int(p_rows, argc, argv, i, "p-rows");
					if (p_rows <= 2) {
						std::cout << "Pattern rows must be > 2" << std::endl;
						p_rows = 6;
					}
				} else if (argv[i] == std::string ("--p-cols")) {
					string_to_int(p_cols, argc, argv, i, "p-cols");
					if (p_cols <= 2) {
						std::cout << "Pattern cols must be > 2" << std::endl;
						p_cols = 6;
					}
				}
			}
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

			while(true) {
				std::cin >> in;
				if (in == "end") {
					break;
				} else if (in == "cl") {
					if (pattern_ims.size())  {
						pattern_ims.pop_back();
					}
				}

				in = path + "/" + in;
				cv::matrixr im;

				try {
					im = cv::imread(in, cv::REAL, 1);
				} catch (std::runtime_error &e) {
					std::cout << "Unexpected error occurred while reading image:\n" << in << std::endl;
					std::cout << "Error message: " << e.what() << std::endl;
					continue;
				}

				if (!im) {
					std::cout << "Failure loading image at: \n" << in << std::endl;
				} else {
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
		} else {
			#pragma omp parallel for
			for (int i = 0; i < im_files.size(); ++i) {
				cv::matrixr im;
				std::string &in = im_files[i];
				try {
					im = cv::imread(in, cv::REAL, 1);
				} catch (std::runtime_error &e) {
					std::cout << "Unexpected error occurred while reading image:\n" << in << std::endl;
					std::cout << "Error message: " << e.what() << std::endl;
					continue;
				}
				if (!im) {
					#pragma omp critical
					std::cout << "Failure loading image at: \n" << in << std::endl;
				} else {
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

					#pragma omp critical
					{
						pattern_ims.push_back(im);
						std::cout << "Image at " << in << " accepted." << std::endl;
					}
				}
			}
		}

		if (pattern_ims.empty()) {
			std::cout << "Pattern data empty - detection exiting..." << std::endl;
			return EXIT_SUCCESS;
		}

		std::string out;
		std::cout << "Where to write detected pattern point data?" << std::endl;
		std::cin >> out;

		std::cout << "=========================================" << std::endl;
		std::cout << "Running pattern detection..." << std::endl;

		pattern_detection(pattern_ims, out, p_rows, p_cols, ang_thresh, mag_thresh, nn_count);

		std::cout << "Pattern detection finished successfully..." << std::endl;
		std::cout << "=========================================" << std::endl;
		return EXIT_SUCCESS;


	} else if (argv[1] == std::string("--zhang")) {
		pattern_file = "zhang";
	} else {
		pattern_file = argv[1];
	}

	if (argc > 2) {
		for (int i = 1; i < argc; ++i) {
			std::string _arg = argv[i];
			if (_arg == "--no-skew") {
				no_skew = true;
			} else if (_arg == "--fixed-aspect") {
				fixed_aspect = true;
			} else if (_arg == "--ftol") {
				string_to_double(ftol, argc, argv, i, "ftol");
			} else if (_arg == "--model-square-size") {
				string_to_double(model_square_size, argc, argv, i, "model-square-size");
			} else if (_arg == "--skip-opt") {
				skip_optimization = true;
			} else if (_arg == "--skip-ext-opt") {
				skip_extrinsic_optmization = true;
			} else if (_arg == "--skip-dist-opt") {
				skip_distortion_optimization = true;
			}
		}
	}

// =================== INITIALIZE CALIBRATION ============================= //

	unsigned im_w, im_h;

	std::vector<std::vector<cv::vec2r>> image_points_nrm;
	std::vector<cv::vec3r> model_points;

	if (pattern_file == "zhang") {
		read_zhang_data("/home/relja/git/camera_calibration/calib_data/zhang_data", image_points_nrm, model_points, im_w, im_h);
	} else {
		if (!read_custom_data(pattern_file, image_points_nrm, model_points, im_w, im_h, model_square_size) ) {
			std::cout << "Error while reading pattern file at: " << pattern_file << std::endl;
			return EXIT_FAILURE;
		}
	}

	auto image_points_orig = image_points_nrm;

	auto N = normalize_image_points(image_points_nrm, im_w, im_h);
	auto N_inv = N.clone();
	cv::invert(N_inv);

	auto image_points_count = image_points_nrm.size();

	std::vector<cv::matrixr> Hs(image_points_count);

	for (unsigned i = 0; i < image_points_count; ++i) {
		ASSERT(image_points_nrm[i].size() == model_points.size());
		Hs[i] = homography_solve(image_points_nrm[i], model_points);
		homography_optimize(image_points_nrm[i], model_points, Hs[i], ftol);
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
	std::vector<std::vector<cv::vec3r>> camera_points(image_points_count);

	std::vector<cv::matrixr> Ks;


	for (unsigned i = 0; i < image_points_count; ++i) {
		auto K = compute_extrinsics(A, N_inv*Hs[i]);
		optimize_extrinsics(image_points_orig[i], model_points, A, K, ftol);

		auto err = calc_reprojection(A, K, model_points, image_points_orig[i], image_points_proj[i]);
		std::cout << "Extrinsics " <<  i << std::endl;
		std::cout << "\nOptimized reprojection error: " << err << std::endl;
		std::cout << "K:\n" << K << std::endl;
		Ks.push_back(K);
	}

	auto k = compute_distortion(image_points_orig, image_points_nrm, image_points_proj, A);
	optimize_distortion(image_points_orig, model_points, A, Ks, k, ftol);
	std::cout << "Init k:\n" << k << std::endl << std::endl;

	if (!skip_optimization)
		optimize_calib(image_points_orig, model_points, A, Ks, k, fixed_aspect, no_skew, ftol);

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

	std::cout << "Mean reprojection error for all patterns: " << (mean_err/image_points_count) << std::endl;

	std::cout << "**********************************************************" << std::endl;

	return EXIT_SUCCESS;
}

