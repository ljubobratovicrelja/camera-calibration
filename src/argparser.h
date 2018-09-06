/*
        Minimalistic argument parsing library.
*/

#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>


/// Type of argument.
enum class arg_type {
    FLAG,     ///< Flag argument, without value.
    STRING,     ///< String argument.
    INTEGER,     ///< Integer argument.
    FLOAT     ///< Flaot argument.
};


/// Argument structure
struct argument {
    bool given;     ///< Tell if argument has been assigned.
    arg_type type;     ///< Type of the argument.
    std::string long_key;     ///< Long key (name) of the argument.
    std::string short_key;     ///< Short key (name) of the argument.
    std::string description;

    union {     // value of the argument, depending on the type.
        bool* flag;
        std::string* str;
        int* integer;
        double* flt;
    };


    argument(bool& flag, const std::string& long_key, const std::string& short_key, const std::string& description);

    argument(std::string& value, const std::string& long_key, const std::string& short_key, const std::string& description);

    argument(int& value, const std::string& long_key, const std::string& short_key, const std::string& description);

    argument(double& value, const std::string& long_key, const std::string& short_key, const std::string& description);

    argument(const argument& copy);
    ~argument();

    argument& operator=(const argument& rhs);

    private:
        void copy(const argument& from);
};


/// Parser of command line arguments.
class arg_parser {
    private:
        std::vector<argument> arguments;

    public:
        arg_parser();
        ~arg_parser();

        /// Define flag argument to this parser.
        arg_parser& def_argument(const argument& arg);

        /// Parse arguments
        void parse(int argc, char** argv);
        std::ostream& print_help(const std::string& greet_message = "", std::ostream& stream = std::cout);

        /// Tell if argument by a key is given.
        bool given(const std::string& key);
        /// Try fetch an argument by a key. If argument is not found, nullptr is returned.
        argument* try_fetch(const std::string& key);
        /// Fetch string argument, assuming such argument is given by a requested key. If there's no such argument,
        // std::invalid_argument is thrown.
        std::string fetch_string(const std::string& key);
        /// Fetch integer argument, assuming such argument is given by a requested key. If there's no such argument,
        // std::invalid_argument is thrown.
        int fetch_int(const std::string& key);
        /// Fetch doubleiing point argument, assuming such argument is given by a requested key. If there's no such argument,
        // std::invalid_argument is thrown.
        double fetch_double(const std::string& key);
};
