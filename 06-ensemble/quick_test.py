"""Quick test runner"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_ensemble import run_all_tests

if __name__ == '__main__':
    run_all_tests()
