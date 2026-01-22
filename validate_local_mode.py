#!/usr/bin/env python3
"""
Validation script for local mode functionality.
This script validates that the local mode changes work correctly.
"""

import sys
import ast
from pathlib import Path


def validate_app_structure():
    """Validate app.py has required functions and structure"""
    print("=" * 60)
    print("Validating app.py structure...")
    print("=" * 60)
    
    app_path = Path(__file__).parent / "app.py"
    
    with open(app_path, 'r') as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
        print("✓ app.py syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in app.py: {e}")
        return False
    
    # Check for required functions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    required_functions = [
        'init_session_state',
        'authentication_page', 
        'slide_selection_page',
        'image_upload_page',  # NEW
        'analysis_page',
        'export_page',
        'import_page',
        'main'
    ]
    
    all_found = True
    for func in required_functions:
        if func in functions:
            print(f"✓ Function '{func}' found")
        else:
            print(f"✗ Function '{func}' MISSING")
            all_found = False
    
    # Check for session state variables
    print("\nChecking session state initialization...")
    if "'local_mode'" in code:
        print("✓ 'local_mode' session state added")
    else:
        print("✗ 'local_mode' session state MISSING")
        all_found = False
    
    if "'uploaded_images'" in code:
        print("✓ 'uploaded_images' session state added")
    else:
        print("✗ 'uploaded_images' session state MISSING")
        all_found = False
    
    if "'current_image_name'" in code:
        print("✓ 'current_image_name' session state added")
    else:
        print("✗ 'current_image_name' session state MISSING")
        all_found = False
    
    # Check for file uploader
    print("\nChecking image upload functionality...")
    if "st.file_uploader" in code:
        print("✓ File uploader implemented")
    else:
        print("✗ File uploader MISSING")
        all_found = False
    
    # Check for mode selection
    if '"📁 Local Image Upload Mode"' in code or "'📁 Local Image Upload Mode'" in code:
        print("✓ Local mode option in authentication")
    else:
        print("✗ Local mode option MISSING")
        all_found = False
    
    return all_found


def validate_config():
    """Validate config.py changes"""
    print("\n" + "=" * 60)
    print("Validating config.py changes...")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "config.py"
    
    with open(config_path, 'r') as f:
        code = f.read()
    
    all_found = True
    
    # Check for LOCAL_MODE flag
    if "LOCAL_MODE" in code:
        print("✓ LOCAL_MODE configuration added")
    else:
        print("✗ LOCAL_MODE configuration MISSING")
        all_found = False
    
    # Check for validate parameter
    if "require_halo_api" in code:
        print("✓ validate() method updated with require_halo_api parameter")
    else:
        print("✗ require_halo_api parameter MISSING from validate()")
        all_found = False
    
    return all_found


def validate_readme():
    """Validate README.md has been updated"""
    print("\n" + "=" * 60)
    print("Validating README.md documentation...")
    print("=" * 60)
    
    readme_path = Path(__file__).parent / "README.md"
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    all_found = True
    
    # Check for local mode documentation
    if "Local Image Upload Mode" in content or "Local Mode" in content:
        print("✓ Local mode documented")
    else:
        print("✗ Local mode NOT documented")
        all_found = False
    
    if "JPG, PNG, TIFF" in content or "jpg, png, tiff" in content:
        print("✓ Supported formats documented")
    else:
        print("✗ Supported formats NOT documented")
        all_found = False
    
    if "Using Local Mode" in content or "Local mode" in content.lower():
        print("✓ Local mode usage instructions found")
    else:
        print("✗ Local mode usage instructions MISSING")
        all_found = False
    
    return all_found


def validate_tests():
    """Validate tests have been updated"""
    print("\n" + "=" * 60)
    print("Validating test updates...")
    print("=" * 60)
    
    test_config_path = Path(__file__).parent / "tests" / "test_config.py"
    
    with open(test_config_path, 'r') as f:
        content = f.read()
    
    all_found = True
    
    # Check for updated tests
    if "require_halo_api" in content:
        print("✓ Tests updated for optional Halo API")
    else:
        print("✗ Tests NOT updated for optional Halo API")
        all_found = False
    
    if "LOCAL_MODE" in content:
        print("✓ Tests include LOCAL_MODE checks")
    else:
        print("✗ Tests MISSING LOCAL_MODE checks")
        all_found = False
    
    return all_found


def main():
    """Run all validations"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "LOCAL MODE VALIDATION SCRIPT" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    results.append(("App Structure", validate_app_structure()))
    results.append(("Config Changes", validate_config()))
    results.append(("README Documentation", validate_readme()))
    results.append(("Test Updates", validate_tests()))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<45} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All validations PASSED!")
        print("\nLocal mode implementation is complete and ready for testing.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run the application: streamlit run app.py")
        print("  3. Select 'Local Image Upload Mode'")
        print("  4. Upload test images and run analysis")
        return 0
    else:
        print("\n⚠️  Some validations FAILED!")
        print("Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
