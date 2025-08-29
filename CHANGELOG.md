# Changelog

All notable changes to this project will be documented in this file.

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