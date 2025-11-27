# Application

This folder contains Python applications that use the CUDA algorithm module.

## Requirements

- Python 3.8 or higher
- Built algorithm module (see `../module/`)

## Usage

1. Build the module first:
   ```bash
   cd ../module
   .\batch\configure.bat
   .\batch\build-debug.bat
   ```

2. Run the Python application:
   ```bash
   python main.py
   ```

## Module Location

The Python script automatically searches for the `algorithm` module in:
- `../module/build/` (and subdirectories)

Make sure the module is built before running the application.

