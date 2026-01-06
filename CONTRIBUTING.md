# Contributing

## Types of Contributions

- **Bug reports**: https://github.com/imewei/xpcsviewer/issues
- **Feature requests**: https://github.com/imewei/xpcsviewer/issues
- **Code contributions**
- **Documentation improvements**

## Development Setup

```bash
# Fork and clone
git clone git@github.com:your_name_here/xpcsviewer.git
cd xpcsviewer

# Install with uv
uv sync
make dev-setup

# Create branch
git checkout -b your-feature-name

# Test changes
make test
make lint

# Commit and push
git commit -m "Description"
git push origin your-feature-name
```

## Pull Request Guidelines

- Include tests for new features
- Update documentation for changes
- Support Python 3.12 and 3.13
- Pass all quality checks

## Code Standards

- Use ruff for linting/formatting
- Add type hints to functions
- Write tests for new code
- Add docstrings to public functions
