#pragma once

/**
 * ISO 8601 timestamp utility with timezone offset.
 */

#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace q3 {

/**
 * Return the current local time as an ISO 8601 string with timezone offset.
 *
 * Example output: "2026-06-18T15:30:45-07:00"
 */
inline std::string get_iso_timestamp_with_timezone() {
    auto current_time = std::chrono::system_clock::now();
    std::time_t calendar_time = std::chrono::system_clock::to_time_t(current_time);
    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(std::localtime(&calendar_time), "%Y-%m-%dT%H:%M:%S%z");
    std::string timestamp_string = timestamp_stream.str();

    if (timestamp_string.length() >= 5) {
        std::string timezone_offset = timestamp_string.substr(timestamp_string.length() - 5);
        if ((timezone_offset[0] == '+' || timezone_offset[0] == '-') &&
            std::isdigit(timezone_offset[1]) && std::isdigit(timezone_offset[2]) &&
            std::isdigit(timezone_offset[3]) && std::isdigit(timezone_offset[4])) {
            timestamp_string = timestamp_string.substr(0, timestamp_string.length() - 2) + ":" +
                               timestamp_string.substr(timestamp_string.length() - 2);
        }
    }
    return timestamp_string;
}

} // namespace q3
