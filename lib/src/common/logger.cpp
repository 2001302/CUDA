#include "common/logger.h"
#include <ctime>

namespace common {

Logger& Logger::get_instance() {
    static Logger instance;
    return instance;
}

std::string Logger::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
#ifdef _WIN32
    struct tm timeinfo;
    localtime_s(&timeinfo, &time);
    ss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
#else
    struct tm timeinfo;
    localtime_r(&time, &timeinfo);
    ss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
#endif
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string Logger::level_to_color(LogLevel level) const {
#ifdef _WIN32
    // Windows console color codes
    switch (level) {
        case LogLevel::DEBUG:   return "";  // Default color
        case LogLevel::INFO:    return "";  // Default color
        case LogLevel::WARNING: return "";  // Default color
        case LogLevel::ERROR:   return "";  // Default color
        default:                return "";
    }
#else
    // ANSI color codes (Linux/Mac)
    switch (level) {
        case LogLevel::DEBUG:   return "\033[36m";  // Cyan
        case LogLevel::INFO:    return "\033[32m";  // Green
        case LogLevel::WARNING: return "\033[33m";  // Yellow
        case LogLevel::ERROR:   return "\033[31m";  // Red
        default:                return "\033[0m";   // Reset
    }
#endif
}

void Logger::log(LogLevel level, const std::string& message) {
    std::string time = get_current_time();
    std::string level_str = level_to_string(level);
    std::string color = level_to_color(level);
    std::string reset = "";
    
#ifndef _WIN32
    reset = "\033[0m";  // ANSI reset code
#endif
    
    std::cout << "[" << time << "] "
              << color << "[" << level_str << "]" << reset
              << " " << message << std::endl;
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

} // namespace common

