#include "argparser.h"

#include <algorithm>
#include <stdexcept>
#include <deque>

static bool flag_help_value;

arg_parser::arg_parser()
{
	// initialize with help flag
	this->def_argument({ flag_help_value, "help", "h", "Get help for this program." });
}

arg_parser::~arg_parser()
{
}

arg_parser & arg_parser::def_argument
(
	const argument& arg
)
{
	this->arguments.push_back(arg);
	return *this;
}

void arg_parser::parse(int argc, char ** argv)
{
	for (int i = 1; i < argc; ++i) {
		auto arg_key = argv[i];
		auto arg = this->try_fetch(arg_key);
		if (!arg) {
			std::cout << "Unrecognized argument given: " << arg_key << std::endl;
			continue;
		}
		arg->given = true;

		if (arg->type == arg_type::FLAG) {
			*arg->flag = true;
			continue;
		}

		// otherwise its a valued type, so next argument must exist
		if (i == argc - 1) {
			throw std::invalid_argument("Given (noted) value argument without value given.");
		}

		std::string arg_value = argv[++i];

		try {
			switch (arg->type) {
			case arg_type::STRING:
				*arg->str = arg_value;
				break;
			case arg_type::INTEGER:
				*arg->integer = std::stoi(arg_value);
				break;
			case arg_type::FLOAT:
				*arg->flt = std::stof(arg_value);
				break;
			}
		}
		catch (std::exception& e) {
			std::string msg = "Failed converting argument value for argument ";
			msg += arg_key;
			throw std::invalid_argument(msg);
		}
	}
}

std::ostream& arg_parser::print_help(const std::string & greet_message, std::ostream & stream)
{
	stream
		<< greet_message << std::endl
		<< "------------" << std::endl;

	for (auto const& arg : this->arguments) {
		stream << "--" << arg.long_key;
		if (!arg.short_key.empty()) {
			stream << "," << arg.short_key;
		}
		if (!arg.description.empty()) {
			stream << ": " << arg.description;
		}
		stream << std::endl;
	}
	
	return stream;
}

bool arg_parser::given(const std::string & key)
{
	if (auto arg = this->try_fetch(key)) {
		return arg->given;
	}
	throw std::invalid_argument("Given key has not been previously introduced to the parser.");
}

argument* arg_parser::try_fetch(const std::string& key) {
	// remove "minus" characters from the key
	std::deque<char> key_d(key.begin(), key.end());
	while (!key_d.empty() && *key_d.begin() == '-') {
		key_d.pop_front();
	}
	std::string key_s(key_d.begin(), key_d.end());
	
	auto search_advocate = [&key_s](const argument& arg) { return arg.long_key == key_s || arg.short_key == key_s; };
	auto res = std::find_if(this->arguments.begin(), this->arguments.end(), search_advocate);
	if (res != this->arguments.end()) {
		return &*res;
	}
	return nullptr;
}

std::string arg_parser::fetch_string(const std::string& key) {
	auto arg = this->try_fetch(key);
	if (arg == nullptr || arg->type != arg_type::STRING) {
		throw std::invalid_argument("Cannot find a string argument by a given key");
	}
	return *arg->str;
}
int arg_parser::fetch_int(const std::string& key) {
	auto arg = this->try_fetch(key);
	if (arg == nullptr || arg->type != arg_type::INTEGER) {
		throw std::invalid_argument("Cannot find an integer argument by a given key");
	}
	return *arg->integer;
}

double arg_parser::fetch_double(const std::string& key) {
	auto arg = this->try_fetch(key);
	if (arg == nullptr || arg->type != arg_type::FLOAT) {
		throw std::invalid_argument("Cannot find a double argument by a given key");
	}
	return *arg->flt;
}

argument::argument
(
	bool& flag,
	const std::string& long_key,
	const std::string& short_key,
	const std::string& description
)
{
	this->given = false;
	this->flag = &flag;
	this->type = arg_type::FLAG;
	this->long_key = long_key;
	this->short_key = short_key;
	this->description = description;
}

argument::argument
(
	std::string& value,
	const std::string& long_key,
	const std::string& short_key,
	const std::string& description
)
{
	this->given = false;
	this->str = &value;
	this->type = arg_type::STRING;
	this->long_key = long_key;
	this->short_key = short_key;
	this->description = description;
}

argument::argument
(
	int& value,
	const std::string& long_key,
	const std::string& short_key,
	const std::string& description
)
{
	this->given = false;
	this->integer = &value;
	this->type = arg_type::INTEGER;
	this->long_key = long_key;
	this->short_key = short_key;
	this->description = description;
}

argument::argument
(
	double& value,
	const std::string& long_key,
	const std::string& short_key,
	const std::string& description
)
{
	this->given = false;
	this->flt = &value;
	this->type = arg_type::FLOAT;
	this->long_key = long_key;
	this->short_key = short_key;
	this->description = description;
}

argument::argument(const argument & copy)
{
	this->copy(copy);
}

argument::~argument()
{
}

argument & argument::operator=(const argument & rhs)
{
	if (this != &rhs) {
		this->copy(rhs);
	}
	return *this;
}

void argument::copy(const argument & from)
{
	this->given = from.given;
	this->type = from.type;
	this->long_key = from.long_key;
	this->short_key = from.short_key;
	this->description = from.description;
	
	switch (this->type) {
	case arg_type::FLAG:
		this->flag = from.flag;
		break;
	case arg_type::STRING:
		this->str = from.str;
		break;
	case arg_type::FLOAT:
		this->flt = from.flt;
		break;
	case arg_type::INTEGER:
		this->integer = from.integer;
		break;
	}
}
