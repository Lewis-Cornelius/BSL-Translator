# Contributing to BSL Translator

Thank you for considering contributing to the BSL Translator project! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues** to see if it's already been reported
2. **Create a new issue** with a clear title and description
3. Include:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)
   - Screenshots or error messages if applicable

### Submitting Changes

1. **Fork the repository**
2. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Test your changes** thoroughly
5. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** with a clear description of your changes

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused on a single task

### Python Development

- Use Python 3.11.9 for compatibility
- Update `requirements.txt` if you add new dependencies
- Test your code before submitting

### Documentation

- Update documentation for any new features
- Keep the README up to date
- Add docstrings to functions and classes

### Testing

Before submitting your changes:

1. Test the ML model with various hand gestures
2. Verify the web interface works correctly
3. Test Firebase integration if you make backend changes
4. Check that the Raspberry Pi code still works (if applicable)

## Project Structure

```
BSL-Translator/
├── project/
│   └── server/          # Backend server and ML code
│       └── website/     # Web interface
├── RaspberryPi/         # Raspberry Pi specific code
├── training_process/    # ML model training scripts
├── Documentation/       # Project documentation
└── old/                 # Deprecated code (for reference)
```

## Areas for Contribution

We welcome contributions in these areas:

- **Model Improvement**: Enhance the ML model accuracy
- **New Gestures**: Add support for more BSL gestures
- **Performance**: Optimize processing speed
- **UI/UX**: Improve the web interface
- **Documentation**: Enhance or translate documentation
- **Testing**: Add unit tests and integration tests
- **Mobile Apps**: Develop mobile applications
- **Accessibility**: Improve accessibility features

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out through GitHub

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on collaboration and learning

Thank you for contributing to making sign language more accessible! 🙌
