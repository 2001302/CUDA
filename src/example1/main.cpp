#include "common/logger.h"

int main() {
    auto& logger = common::Logger::get_instance();
    
    logger.info("Hello, World!");
    logger.debug("This is a debug message");
    logger.warning("This is a warning message");
    logger.error("This is an error message");
    
    return 0;
}
