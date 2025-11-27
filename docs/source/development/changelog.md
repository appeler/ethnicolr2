# Changelog

For the complete changelog, see the [CHANGELOG.md](https://github.com/appeler/ethnicolr2/blob/main/CHANGELOG.md) file in the repository root.

## Recent Releases

### Version 0.2.0 (2025-11-27)

Major modernization release:

- **Fixed NumPy 2.x compatibility** - Resolved dependency version conflicts
- **Migrated to uv** - Modern dependency management
- **Documentation overhaul** - Converted from RST to Markdown  
- **Improved project structure** - Tests moved to root directory
- **All tests now pass** - Fixed 29 failing tests

### Version 0.1.4 (2025-09-03)

Type safety and error handling improvements:

- **Added comprehensive type hints** throughout codebase
- **Enhanced error handling** with specific exception types
- **Improved documentation** with better docstrings
- **Robust input validation** across all functions

### Previous Versions

See the [full changelog](https://github.com/appeler/ethnicolr2/blob/main/CHANGELOG.md) for complete version history.

## Breaking Changes

### Version 0.2.0

- **NumPy requirement**: Minimum version increased to 1.25.2
- **Python requirement**: Now requires Python 3.11-3.13
- **Development workflow**: Now uses `uv` instead of pip for development

## Migration Guide

### Upgrading to 0.2.0

```bash
# Update to latest version
pip install --upgrade ethnicolr2

# Or use uv
uv add ethnicolr2
```

If you encounter NumPy compatibility issues:

```bash
pip install --upgrade "numpy>=2.3.5"
```

### For Developers

```bash
# Old way (still works)
pip install -e ".[dev,test,docs]"

# New way (recommended)
uv sync --all-groups
```