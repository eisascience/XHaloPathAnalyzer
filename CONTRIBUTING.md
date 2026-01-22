# Contributing to XHalo Path Analyzer

Thank you for your interest in contributing to XHalo Path Analyzer! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/eisascience/XHaloPathAnalyzer/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear use case description
   - Expected behavior
   - Why this feature would be useful
   - Possible implementation approach (optional)

### Contributing Code

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/ -v
   python examples/basic_usage.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: your feature description"
   ```
   
   Use conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Wait for review

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for functions and classes

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception is raised
    """
    pass
```

### Code Organization

- Keep functions focused and small
- Use meaningful variable names
- Add comments for complex logic
- Group related functionality

### Testing

- Write unit tests for new functions
- Maintain or improve test coverage
- Test edge cases
- Use pytest fixtures for setup

Example test:
```python
def test_function_name():
    """Test description"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Development Setup

1. **Clone and install in development mode**
   ```bash
   git clone https://github.com/eisascience/XHaloPathAnalyzer.git
   cd XHaloPathAnalyzer
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks (optional)**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=xhalo
   ```

## Project Structure

```
XHaloPathAnalyzer/
├── xhalo/              # Main package
│   ├── api/           # Halo API integration
│   ├── ml/            # ML models
│   ├── ui/            # UI components
│   └── utils/         # Utility functions
├── tests/             # Unit tests
├── examples/          # Example scripts
├── docs/              # Documentation
└── app.py            # Main Streamlit app
```

## Documentation

- Update README.md for major changes
- Add docstrings to new functions
- Update relevant documentation files
- Include usage examples

## Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged

## Areas to Contribute

### High Priority
- Additional ML model integrations
- Performance optimizations
- Enhanced visualization options
- More comprehensive tests

### Good First Issues
- Documentation improvements
- Bug fixes
- Example scripts
- UI enhancements

### Advanced Contributions
- Custom model architectures
- Advanced image processing algorithms
- Integration with other pathology platforms
- Scalability improvements

## Questions?

- Open a discussion on GitHub
- Comment on relevant issues
- Join our community channels

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to XHalo Path Analyzer! 🎉
