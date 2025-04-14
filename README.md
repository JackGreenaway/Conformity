# Conformity

Conformity is a Python library designed to provide conformal prediction tools for regression and classification tasks. It ensures reliable uncertainty quantification in machine learning models.

## Prerequisites

Before using this repository, ensure you have the following installed:
- [Python 3.11](https://www.python.org/downloads/)
- [UV](https://uv.pm/) (for package management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/conformity.git
   cd conformity
   ```

2. Install dependencies using `UV`:
   ```bash
   uv install
   ```

## Usage

### Running the Project
To use the library, you can import the modules and classes in your Python scripts. For example:
```python
from src.conformity.regression import ConformalRegressor
from src.conformity.classifier import ConformalClassifier
```

### Running Tests
To execute tests (if available), use:
```bash
uv test
```

### Development
For development purposes, install the `dev` dependencies:
```bash
uv install --group dev
```

### Building the Project
To build the project for distribution:
```bash
uv build
```

## Features

- **Conformal Regressor**: Provides prediction intervals for regression tasks.
- **Conformal Classifier**: Generates prediction sets for classification tasks.
- **Metrics**: Includes utilities to evaluate coverage and efficiency of prediction intervals and sets.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, feel free to open an issue in the repository or contact the maintainers.
