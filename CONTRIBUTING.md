# Contributing to AI Case Studies

Thank you for your interest in contributing to this repository! This guide will help you get started.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest enhancements
- Provide clear descriptions and steps to reproduce
- Include relevant code snippets or error messages

### Adding New Case Studies

1. **Follow the Template Structure**
   - Use the template in `templates/case-study-template.md`
   - Ensure all required components are present
   - Maintain consistency with existing case studies

2. **Code Quality Standards**
   - Follow PEP 8 style guidelines
   - Include type hints for all functions
   - Write comprehensive docstrings (Google style)
   - Add inline comments for complex logic
   - Ensure >80% test coverage

3. **Documentation Requirements**
   - Each case study must include:
     - README.md with overview
     - problem_statement.md with business context
     - solution_approach.md with architecture
     - evaluation.md with metrics and results
     - trade_offs.md with design decisions
     - interview_questions.md with Q&A
   - Use clear, professional language
   - Include diagrams where helpful

4. **Code Organization**
   ```
   your-case-study/
   ├── README.md
   ├── problem_statement.md
   ├── solution_approach.md
   ├── data/
   ├── notebooks/
   ├── src/
   ├── tests/
   ├── evaluation.md
   ├── trade_offs.md
   ├── interview_questions.md
   └── requirements.txt
   ```

5. **Data Guidelines**
   - Keep sample datasets small (<10MB)
   - Use synthetic or publicly available data
   - Include data_description.md
   - Never include proprietary or sensitive data
   - Provide data generation scripts if needed

6. **Testing Requirements**
   - Write unit tests for all functions
   - Use pytest framework
   - Achieve >80% code coverage
   - Include edge cases
   - Test with sample data

### Pull Request Process

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ai-usecases.git
   cd ai-usecases
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-case-study-name
   ```

3. **Make Your Changes**
   - Follow the structure and guidelines above
   - Test thoroughly
   - Update documentation

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: [Category] Your Case Study Name"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-case-study-name
   ```
   - Create a Pull Request on GitHub
   - Provide a clear description
   - Reference any related issues

6. **Code Review**
   - Address reviewer feedback
   - Make requested changes
   - Ensure CI/CD checks pass

### Code Style Guide

#### Python Code
```python
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    columns: List[str],
    fill_na: bool = True
) -> pd.DataFrame:
    """
    Preprocess input dataframe by handling missing values and selecting columns.

    Args:
        df: Input dataframe to preprocess
        columns: List of column names to select
        fill_na: Whether to fill missing values with mean

    Returns:
        Preprocessed dataframe with selected columns

    Raises:
        ValueError: If columns are not found in dataframe
    """
    if not all(col in df.columns for col in columns):
        raise ValueError("Some columns not found in dataframe")

    result = df[columns].copy()

    if fill_na:
        result = result.fillna(result.mean())

    return result
```

#### Docstring Format (Google Style)
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Provide type information
- Add examples for complex functions

#### Testing
```python
import pytest
import pandas as pd
from src.data_preprocessing import preprocess_data


def test_preprocess_data_basic():
    """Test basic preprocessing functionality."""
    df = pd.DataFrame({
        'a': [1, 2, None],
        'b': [4, 5, 6]
    })
    result = preprocess_data(df, ['a', 'b'], fill_na=True)
    assert result.shape == (3, 2)
    assert not result['a'].isna().any()


def test_preprocess_data_missing_columns():
    """Test that ValueError is raised for missing columns."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(ValueError):
        preprocess_data(df, ['a', 'missing_col'])
```

### Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: [Category] Case study name` - New case study
- `Update: [Category] Improve documentation` - Documentation updates
- `Fix: [Category] Correct model implementation` - Bug fixes
- `Refactor: [Category] Simplify preprocessing` - Code refactoring
- `Test: [Category] Add unit tests` - Test additions

### Review Criteria

Pull requests will be reviewed for:

1. **Correctness**
   - Code runs without errors
   - Tests pass
   - ML implementation is sound

2. **Code Quality**
   - Follows style guidelines
   - Proper type hints and docstrings
   - Clean, readable code

3. **Documentation**
   - Complete and clear
   - All required files present
   - Proper markdown formatting

4. **Educational Value**
   - Relevant to interviews
   - Clear learning objectives
   - Good interview questions

5. **Reproducibility**
   - Clear setup instructions
   - All dependencies listed
   - Sample data provided

## Questions?

If you have questions about contributing:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation in `docs/`

Thank you for contributing!
