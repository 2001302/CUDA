#include "common/Logger.h"
#include <ctime>

namespace common {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

std::string Logger::getCurrentTime() const {
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

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string Logger::levelToColor(LogLevel level) const {
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
    std::string time = getCurrentTime();
    std::string levelStr = levelToString(level);
    std::string color = levelToColor(level);
    std::string reset = "";
    
#ifndef _WIN32
    reset = "\033[0m";  // ANSI reset code
#endif
    
    std::cout << "[" << time << "] "
              << color << "[" << levelStr << "]" << reset
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

