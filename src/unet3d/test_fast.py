"""Run tests in groups to identify issues more easily"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
from test_unet3d import (
    TestDoubleConv3D, TestEncoderBlock, TestDecoderBlock,
    TestUNet3DNet, TestUNet3DModel, TestAPIConsistency
)

print("=" * 80)
print("RUNNING ARCHITECTURE TESTS (Fast)")
print("=" * 80)

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Only add fast tests
suite.addTests(loader.loadTestsFromTestCase(TestDoubleConv3D))
suite.addTests(loader.loadTestsFromTestCase(TestEncoderBlock))
suite.addTests(loader.loadTestsFromTestCase(TestDecoderBlock))
suite.addTests(loader.loadTestsFromTestCase(TestUNet3DNet))
suite.addTests(loader.loadTestsFromTestCase(TestUNet3DModel))
suite.addTests(loader.loadTestsFromTestCase(TestAPIConsistency))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("\n" + "=" * 80)
print(f"Tests run: {result.testsRun}")
print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
print("=" * 80)

sys.exit(0 if result.wasSuccessful() else 1)
