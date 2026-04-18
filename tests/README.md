# Test Suite

This directory contains tests for the NIFTY 50 Stock Lab system.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage report
pytest --cov=backend --cov=frontend --cov-report=html
# Coverage report: htmlcov/index.html

# Run specific test class
pytest tests/test_utils.py::TestTickerValidation -v

# Run specific test function
pytest tests/test_utils.py::TestTickerValidation::test_nifty50_ticker_recognized -v
```

## Test Organization

- **test_utils.py** - Unit tests for utility functions, ticker validation, data normalization
- **test_api.py** - API endpoint tests, request validation, response schema checks
- **test_integration.py** - (Optional) End-to-end integration tests

## Writing Tests

### Basic Unit Test

```python
import pytest
from backend.utils import get_ticker_sector

class TestMyFeature:
    def test_basic_case(self):
        """Test description - what should happen."""
        result = get_ticker_sector("RELIANCE.NS")
        assert result is not None
```

### Parametrized Tests (Test Multiple Cases)

```python
@pytest.mark.parametrize(
    "ticker,expected_sector",
    [
        ("RELIANCE.NS", "Energy"),
        ("INFY.NS", "Information Technology"),
        ("INVALID.NS", None),
    ]
)
def test_ticker_sectors(ticker, expected_sector):
    result = get_ticker_sector(ticker)
    assert result == expected_sector
```

### Testing Exceptions

```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError):
        process_data(None)  # Should raise ValueError
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Shared test data."""
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'close': [100, 102],
    })

def test_with_fixture(sample_data):
    assert len(sample_data) == 2
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('backend.data_ingestion.yfinance.download')
def test_with_mock(mock_download):
    mock_download.return_value = pd.DataFrame({...})
    result = fetch_data("RELIANCE.NS")
    assert result is not None
```

## Example Topics

### Testing Data Ingestion
- ✓ test_utils.py - Ticker validation
- TODO - test_data_ingestion.py: Date parsing, deduplication, multi-source loading

### Testing Predictions
- TODO - test_predictions.py: Model training, inference, fallback handling

### Testing API Routes
- Partial - test_api.py: Endpoint examples (mostly commented for fixture setup)

### Testing Feature Engineering
- TODO - test_features.py: Feature generation, NaN handling, scaling

## Best Practices

1. **Test one thing per test** - Avoid testing multiple unrelated behaviors
2. **Use descriptive names** - `test_ticker_case_insensitive` is better than `test_1`
3. **Keep tests independent** - Each test should be runnable in any order
4. **Don't mock too much** - Test real behavior when possible; mock external services
5. **Use fixtures for shared data** - Helps DRY up test code
6. **Test both success and failure** - Check happy path and error cases
7. **Add type hints** - Helps with IDE support: `def test_feature() -> None:`

## CI/CD Integration

Tests run automatically on:
- Every push to `develop` or `main` branches
- Every pull request
- Across Python 3.10, 3.11, 3.12

See `.github/workflows/ci-cd.yml` for configuration.

## Coverage Goals

- Aim for 80%+ coverage on critical modules (data, models, predictions)
- 60%+ on peripheral modules (utils, display)
- Focus on branches & edge cases, not just lines

Track coverage with:
```bash
pytest --cov=backend --cov=frontend --cov-report=term-missing
```

## Troubleshooting

**Q: Tests fail with "import not found"**
A: Ensure you're running from project root: `cd nifty50_prediction_system && pytest`

**Q: Tests pass locally but fail in CI**
A: Check Python version (CI runs 3.10, 3.11, 3.12). May be version-specific issues.

**Q: Mock not working as expected**
A: Ensure you're patching at the location where it's used, not where it's defined.

**Q: Fixtures not accessible in other files**
A: Create `conftest.py` in tests/ directory to share fixtures across files.

## Next Steps

1. Implement tests incrementally as features are added
2. Aim for critical path coverage first (data → predictions → API)
3. Add integration tests for multi-step workflows
4. Set up coverage reports in CI/CD
5. Celebrate reaching 80% coverage! 🎉

For more on pytest, see: https://docs.pytest.org
