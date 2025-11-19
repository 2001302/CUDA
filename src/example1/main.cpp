#include "common/Logger.h"

int main() {
    auto& logger = common::Logger::getInstance();
    
    logger.info("Hello, World!");
    logger.debug("This is a debug message");
    logger.warning("This is a warning message");
    logger.error("This is an error message");
    
    return 0;
}

