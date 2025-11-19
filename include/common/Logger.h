#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace common {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance();
    
    void log(LogLevel level, const std::string& message);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string getCurrentTime() const;
    std::string levelToString(LogLevel level) const;
    std::string levelToColor(LogLevel level) const;
};

} // namespace common

#endif // LOGGER_H

