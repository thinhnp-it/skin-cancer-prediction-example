"""
Setup Verification Script
Run this script to verify your environment is properly configured
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 11:
        print("  ✓ Python 3.11+ detected")
        return True
    else:
        print("  ⚠️  Warning: Python 3.11+ recommended")
        return False


def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'jupyter'
    ]

    print("\nChecking required packages:")
    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False

    return all_installed


def check_directory_structure():
    """Check if project directories exist"""
    print("\nChecking directory structure:")
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'models',
        'results',
        'docs'
    ]

    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ - MISSING")
            all_exist = False

    return all_exist


def check_dataset():
    """Check if dataset is downloaded"""
    print("\nChecking dataset:")
    dataset_path = 'data/HAM10000_metadata.csv'

    if os.path.exists(dataset_path):
        file_size = os.path.getsize(dataset_path) / 1024  # KB
        print(f"  ✓ HAM10000_metadata.csv found ({file_size:.1f} KB)")
        return True
    else:
        print(f"  ✗ HAM10000_metadata.csv NOT FOUND")
        print(f"     Please download from: https://www.kaggle.com/datasets/kiaskhoshdast/ham10000-metadatacsv")
        print(f"     Place in: {os.path.abspath(dataset_path)}")
        return False


def check_source_files():
    """Check if source files exist"""
    print("\nChecking source files:")
    source_files = [
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessing.py'
    ]

    all_exist = True
    for file in source_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False

    return all_exist


def check_documentation():
    """Check if documentation files exist"""
    print("\nChecking documentation:")
    doc_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'docs/TASK_01_DATA_EXPLORATION.md',
        'docs/TASK_02_DATA_PREPROCESSING.md',
        'docs/TASK_03_BINARY_CLASSIFICATION.md',
        'docs/TASK_04_MULTICLASS_CLASSIFICATION.md'
    ]

    all_exist = True
    for file in doc_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Main verification function"""
    print("=" * 60)
    print("SKIN CANCER DETECTION - SETUP VERIFICATION")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_packages()),
        ("Directory Structure", check_directory_structure()),
        ("Dataset", check_dataset()),
        ("Source Files", check_source_files()),
        ("Documentation", check_documentation())
    ]

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - READY TO START!")
        print("\nNext steps:")
        print("1. Activate virtual environment: source venv/bin/activate")
        print("2. Start Jupyter: jupyter notebook")
        print("3. Open: notebooks/01_data_exploration.ipynb")
    else:
        print("⚠️  SOME CHECKS FAILED - PLEASE FIX BEFORE STARTING")
        print("\nRefer to QUICKSTART.md for setup instructions")

    print("=" * 60)


if __name__ == "__main__":
    main()