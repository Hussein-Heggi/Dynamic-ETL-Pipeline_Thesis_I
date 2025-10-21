"""
Test script for custom feature execution with RestrictedPython.
"""

import json

import numpy as np
import pandas as pd
import yaml

from dsl_validator import validate_dsl
from enrichment import apply_features

# Track test results
test_results = {
    "passed": 0,
    "failed": 0,
    "total": 0
}


def record_test_result(test_name, passed):
    """Record test result"""
    test_results["total"] += 1
    if passed:
        test_results["passed"] += 1
        print(f"✓ {test_name}: PASSED")
    else:
        test_results["failed"] += 1
        print(f"✗ {test_name}: FAILED")


# Create sample data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="1h")
df = pd.DataFrame(
    {
        "ticker": ["AAPL"] * 50 + ["GOOGL"] * 50,
        "ts": list(dates[:50]) + list(dates[:50]),
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(100, 200, 100),
        "low": np.random.uniform(100, 200, 100),
        "close": np.random.uniform(100, 200, 100),
        "volume": np.random.uniform(1000000, 5000000, 100),
    }
)

# Ensure high is the highest and low is the lowest
df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

print("Sample DataFrame:")
print(df.head())
print()

# Load registry
with open("registry.yaml", "r") as f:
    registry = yaml.safe_load(f)

# Test 1: Simple single-line custom feature
print("=" * 80)
print("Test 1: Simple single-line custom feature (price-to-open ratio)")
print("=" * 80)

dsl_string_1 = json.dumps(
    {
        "features": [
            {
                "name": "custom_price_ratio",
                "params": {
                    "code": "series = g['close'] / g['open']",
                    "as": "close_open_ratio",
                },
            }
        ]
    }
)

dsl_1, errors_1 = validate_dsl(dsl_string_1, registry)
try:
    assert not errors_1, f"Validation failed: {errors_1}"
    df_result_1 = apply_features(df.copy(), dsl_1, registry)
    assert "close_open_ratio" in df_result_1.columns, "Output column missing"
    print(f"DSL: {dsl_1}")
    print(f"\nNew columns added: {[c for c in df_result_1.columns if c not in df.columns]}")
    print("\nSample results:")
    print(df_result_1[["ticker", "close", "open", "close_open_ratio"]].head(10))
    record_test_result("Test 1 - Simple custom feature", True)
except Exception as e:
    print(f"Error: {e}")
    record_test_result("Test 1 - Simple custom feature", False)
print()

# Test 2: Multi-line custom feature
print("=" * 80)
print("Test 2: Multi-line custom feature (normalized momentum)")
print("=" * 80)

dsl_string_2 = json.dumps(
    {
        "features": [
            {
                "name": "custom_norm_momentum",
                "params": {
                    "code": "momentum = g['close'] - g['close'].shift(5)\nrolling_std = g['close'].rolling(10).std()\nseries = momentum / rolling_std",
                    "as": "normalized_momentum_5_10",
                },
            }
        ]
    }
)

dsl_2, errors_2 = validate_dsl(dsl_string_2, registry)
try:
    assert not errors_2, f"Validation failed: {errors_2}"
    df_result_2 = apply_features(df.copy(), dsl_2, registry)
    assert "normalized_momentum_5_10" in df_result_2.columns, "Output column missing"
    print(f"DSL: {dsl_2}")
    print(f"\nNew columns added: {[c for c in df_result_2.columns if c not in df.columns]}")
    print("\nSample results:")
    print(df_result_2[["ticker", "close", "normalized_momentum_5_10"]].head(15))
    record_test_result("Test 2 - Multiline custom feature", True)
except Exception as e:
    print(f"Error: {e}")
    record_test_result("Test 2 - Multiline custom feature", False)
print()

# Test 3: Mix of standard and custom features
print("=" * 80)
print("Test 3: Mix of standard and custom features")
print("=" * 80)

dsl_string_3 = json.dumps(
    {
        "features": [
            {"name": "sma", "params": {"on": "close", "window": 10}},
            {
                "name": "custom_midpoint",
                "params": {
                    "code": "series = (g['high'] + g['low']) / 2",
                    "as": "hl_midpoint",
                },
            },
            {"name": "rsi", "params": {"on": "close", "window": 14}},
        ]
    }
)

dsl_3, errors_3 = validate_dsl(dsl_string_3, registry)
try:
    assert not errors_3, f"Validation failed: {errors_3}"
    df_result_3 = apply_features(df.copy(), dsl_3, registry)
    assert all(col in df_result_3.columns for col in ["sma_close_10", "hl_midpoint", "rsi_close_14"]), "Missing columns"
    print(f"DSL: {dsl_3}")
    print(f"\nNew columns added: {[c for c in df_result_3.columns if c not in df.columns]}")
    print("\nSample results:")
    print(df_result_3[["ticker", "close", "high", "low", "sma_close_10", "hl_midpoint", "rsi_close_14"]].head(15))
    record_test_result("Test 3 - Mix of standard and custom features", True)
except Exception as e:
    print(f"Error: {e}")
    record_test_result("Test 3 - Mix of standard and custom features", False)
print()

# Test 4: Invalid custom feature (should fail validation)
print("=" * 80)
print("Test 4: Invalid custom feature - syntax error")
print("=" * 80)

dsl_string_4 = json.dumps(
    {
        "features": [
            {
                "name": "custom_bad_syntax",
                "params": {
                    "code": "series = g['close'] / ",  # Syntax error
                    "as": "bad_feature",
                },
            }
        ]
    }
)

dsl_4, errors_4 = validate_dsl(dsl_string_4, registry)
try:
    assert errors_4, "Expected validation to fail but it passed"
    print("Validation errors (expected):")
    for error in errors_4:
        print(f"  - {error}")
    record_test_result("Test 4 - Syntax error detection", True)
except AssertionError as e:
    print(f"Error: {e}")
    record_test_result("Test 4 - Syntax error detection", False)
print()

# Test 5: Custom feature without 'series' assignment (should fail at runtime)
print("=" * 80)
print("Test 5: Custom feature missing 'series' assignment")
print("=" * 80)

dsl_string_5 = json.dumps(
    {
        "features": [
            {
                "name": "custom_no_series",
                "params": {
                    "code": "result = g['close'] / g['open']",  # Wrong variable name
                    "as": "wrong_var",
                },
            }
        ]
    }
)

dsl_5, errors_5 = validate_dsl(dsl_string_5, registry)
try:
    assert not errors_5, f"Validation failed unexpectedly: {errors_5}"
    print("Attempting to apply features (should fail at runtime):")
    try:
        df_result_5 = apply_features(df.copy(), dsl_5, registry)
        print("✗ UNEXPECTED: Feature application succeeded (should have failed)")
        record_test_result("Test 5 - Missing 'series' variable detection", False)
    except ValueError as e:
        print(f"✓ EXPECTED ERROR: {e}")
        record_test_result("Test 5 - Missing 'series' variable detection", True)
except Exception as e:
    print(f"Error: {e}")
    record_test_result("Test 5 - Missing 'series' variable detection", False)
print()

# ============================================================================
# SECURITY TESTS - Ensure RestrictedPython blocks malicious operations
# ============================================================================

# Test 6: Attempt to open a file
print("=" * 80)
print("Test 6: Security - Attempt to open a file (should fail)")
print("=" * 80)

dsl_string_6 = json.dumps(
    {
        "features": [
            {
                "name": "custom_file_access",
                "params": {
                    "code": "with open('/etc/passwd', 'r') as f:\n    content = f.read()\nseries = g['close']",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_6, errors_6 = validate_dsl(dsl_string_6, registry)
try:
    assert not errors_6, f"Validation failed: {errors_6}"
    try:
        df_result_6 = apply_features(df.copy(), dsl_6, registry)
        print("✗ FAILED: File access was allowed (security breach!)")
        record_test_result("Test 6 - Block file access", False)
    except (RuntimeError, NameError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 6 - Block file access", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 6 - Block file access", False)
print()

# Test 7: Attempt to import os
print("=" * 80)
print("Test 7: Security - Attempt to import os (should fail)")
print("=" * 80)

dsl_string_7 = json.dumps(
    {
        "features": [
            {
                "name": "custom_import_os",
                "params": {
                    "code": "import os\nseries = g['close']",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_7, errors_7 = validate_dsl(dsl_string_7, registry)
try:
    assert not errors_7, f"Validation failed: {errors_7}"
    try:
        df_result_7 = apply_features(df.copy(), dsl_7, registry)
        print("✗ FAILED: Import was allowed (security breach!)")
        record_test_result("Test 7 - Block imports", False)
    except (RuntimeError, ImportError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 7 - Block imports", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 7 - Block imports", False)
print()

# Test 8: Attempt to use eval
print("=" * 80)
print("Test 8: Security - Attempt to use eval (should fail)")
print("=" * 80)

dsl_string_8 = json.dumps(
    {
        "features": [
            {
                "name": "custom_eval",
                "params": {
                    "code": "series = eval('g[\\'close\\']')",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_8, errors_8 = validate_dsl(dsl_string_8, registry)
try:
    assert not errors_8, f"Validation failed: {errors_8}"
    try:
        df_result_8 = apply_features(df.copy(), dsl_8, registry)
        print("✗ FAILED: eval() was allowed (security breach!)")
        record_test_result("Test 8 - Block eval", False)
    except (RuntimeError, NameError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 8 - Block eval", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 8 - Block eval", False)
print()

# Test 9: Attempt to use exec
print("=" * 80)
print("Test 9: Security - Attempt to use exec (should fail)")
print("=" * 80)

dsl_string_9 = json.dumps(
    {
        "features": [
            {
                "name": "custom_exec",
                "params": {
                    "code": "exec('series = g[\\'close\\']')",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_9, errors_9 = validate_dsl(dsl_string_9, registry)
try:
    assert not errors_9, f"Validation failed: {errors_9}"
    try:
        df_result_9 = apply_features(df.copy(), dsl_9, registry)
        print("✗ FAILED: exec() was allowed (security breach!)")
        record_test_result("Test 9 - Block exec", False)
    except (RuntimeError, NameError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 9 - Block exec", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 9 - Block exec", False)
print()

# Test 10: Attempt to list directory
print("=" * 80)
print("Test 10: Security - Attempt to list directory (should fail)")
print("=" * 80)

dsl_string_10 = json.dumps(
    {
        "features": [
            {
                "name": "custom_listdir",
                "params": {
                    "code": "import os\nfiles = os.listdir('.')\nseries = g['close']",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_10, errors_10 = validate_dsl(dsl_string_10, registry)
try:
    assert not errors_10, f"Validation failed: {errors_10}"
    try:
        df_result_10 = apply_features(df.copy(), dsl_10, registry)
        print("✗ FAILED: Directory listing was allowed (security breach!)")
        record_test_result("Test 10 - Block directory listing", False)
    except (RuntimeError, ImportError, NameError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 10 - Block directory listing", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 10 - Block directory listing", False)
print()

# Test 11: Attempt to access __builtins__
print("=" * 80)
print("Test 11: Security - Attempt to access __builtins__ (should fail)")
print("=" * 80)

dsl_string_11 = json.dumps(
    {
        "features": [
            {
                "name": "custom_builtins",
                "params": {
                    "code": "b = __builtins__['open']\nseries = g['close']",
                    "as": "malicious",
                },
            }
        ]
    }
)

dsl_11, errors_11 = validate_dsl(dsl_string_11, registry)
try:
    assert not errors_11, f"Validation failed: {errors_11}"
    try:
        df_result_11 = apply_features(df.copy(), dsl_11, registry)
        print("✗ FAILED: Access to __builtins__ was allowed (security breach!)")
        record_test_result("Test 11 - Block __builtins__ access", False)
    except (RuntimeError, TypeError, KeyError) as e:
        print(f"✓ BLOCKED: {e}")
        record_test_result("Test 11 - Block __builtins__ access", True)
except Exception as e:
    print(f"Error during validation: {e}")
    record_test_result("Test 11 - Block __builtins__ access", False)
print()

# ============================================================================
# FINAL TEST SUMMARY
# ============================================================================

print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total tests: {test_results['total']}")
print(f"Passed: {test_results['passed']} ✓")
print(f"Failed: {test_results['failed']} ✗")
print("=" * 80)

if test_results['failed'] == 0:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️  {test_results['failed']} TEST(S) FAILED")
print("=" * 80)
