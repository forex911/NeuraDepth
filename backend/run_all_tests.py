"""
Test runner for all Task 3.1 related tests
"""
import subprocess
import sys

tests = [
    'test_dataclasses.py',
    'test_extract_scale_features.py',
    'test_compute_adaptive_weights.py',
    'test_adaptive_weights_integration.py'
]

print("="*70)
print("RUNNING ALL TESTS FOR TASK 3.1")
print("="*70)

results = []
for test in tests:
    print(f"\nRunning {test}...")
    result = subprocess.run([sys.executable, test], capture_output=True, text=True)
    passed = result.returncode == 0
    results.append((test, passed))
    
    if not passed:
        print(f"FAILED: {test}")
        print(result.stdout)
        print(result.stderr)

print("\n" + "="*70)
print("FINAL TEST SUMMARY")
print("="*70)
for test, passed in results:
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"{status}: {test}")

passed_count = sum(1 for _, p in results if p)
total_count = len(results)
print("="*70)
print(f"Total: {passed_count}/{total_count} test files passed")
print("="*70)

sys.exit(0 if passed_count == total_count else 1)
