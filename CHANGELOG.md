# Changelog

All notable changes to this project will be documented in this file.

## [0.1.4] - 2025-09-03

### Added
- **Type Safety**: Comprehensive type hints added throughout entire codebase
- **Enhanced Error Handling**: Specific exception types (ValueError, TypeError) replace generic Exception
- **Improved Documentation**: Enhanced docstrings with Args/Returns/Raises sections
- **Input Validation**: Robust type and value checking with helpful error messages

### Improved
- **Developer Experience**: Better IDE support with comprehensive type annotations
- **Error Messages**: More specific and actionable error feedback for users
- **Code Maintainability**: Self-documenting code with clear type expectations
- **API Robustness**: Enhanced parameter validation across all functions

### Technical Details
- Added typing imports to all core modules (ethnicolr_class.py, models.py, dataset.py, utils.py)
- Enhanced lineToTensor method with type checking and better error handling
- Improved LSTM model with parameter validation and type safety
- Updated EthniDataset with comprehensive input validation
- Enhanced all prediction functions with type hints and error handling
- Backward compatible - no breaking changes to existing API

## [0.1.3] - 2024-08-29

### Improved
- **Test Coverage**: Dramatically improved test coverage from 83% to 92%
- **Comprehensive Validation Tests**: Added extensive tests for input validation and error handling
- **Bug Fix Verification**: Added tests to verify all v0.1.2 bug fixes and prevent regressions
- **CLI Testing**: Added command-line interface testing for all main functions
- **Edge Case Handling**: Added tests for empty dataframes, missing values, and data integrity
- **Code Quality**: Enhanced argument parsing with better flexibility and error messages

### Technical Details
- Added 24 new test cases covering critical functionality
- All major bug fixes from v0.1.2 now have dedicated regression tests
- Improved utils.py argument parsing with optional parameters
- Better test isolation and cleanup procedures

## [0.1.2] - 2024-08-29

### Fixed
- **Critical Bug Fix**: Fixed incorrect method reference in census prediction model (`pred_cen_ln_lstm.py`) that was causing it to use Florida model instead of census model
- **Runtime Error Fix**: Added missing `census_year` class variable initialization in `census_ln.py` to prevent AttributeError
- **Data Integrity Fix**: Removed `shuffle=True` from DataLoader during inference to maintain proper order correspondence between input and predictions
- **Input Validation**: Added comprehensive column validation in `pred_fl_full_name` method with clear error messages

### Improved
- **Modernized Dependencies**: Updated deprecated `pkg_resources` imports to use modern `importlib.resources` with backward compatibility fallback
- **Code Organization**: Created constants for model parameters (max name lengths, batch size, hidden size, num layers) to eliminate magic numbers
- **Documentation**: Fixed incorrect docstrings and CLI help text that referenced wrong models
- **Code Quality**: Applied black code formatting across the entire codebase

### Technical Details
- All models now correctly use their intended training data and parameters
- Input validation prevents runtime errors when columns are missing
- Resource loading is future-proof and compatible with modern Python versions
- Prediction order is now guaranteed to match input order

## [0.1.1] - Previous Release
- Initial release with basic functionality