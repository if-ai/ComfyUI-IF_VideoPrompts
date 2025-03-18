#!/usr/bin/env python3
import subprocess
import sys
import os
import pkg_resources
import re

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def is_compatible_versions(transformers_version, autoawq_version):
    # Define compatibility rules
    if transformers_version and autoawq_version:
        # Check for specific compatible combinations
        transformers_major, transformers_minor = map(int, transformers_version.split('.')[:2])
        autoawq_major, autoawq_minor = map(int, autoawq_version.split('.')[:2])
        
        # Requires transformers 4.49.0 or above with autoawq 0.2.8
        if autoawq_major == 0 and autoawq_minor >= 8 and transformers_major == 4 and transformers_minor >= 49:
            return True
    
    return False

def install_dependencies():
    print("Installing dependencies for ComfyUI-IF_VideoPrompts...")
    
    # Check existing package versions
    transformers_version = get_installed_version("transformers")
    autoawq_version = get_installed_version("autoawq")
    
    print(f"Current versions: transformers={transformers_version}, autoawq={autoawq_version}")
    
    # Check if versions are compatible
    if transformers_version and autoawq_version:
        if is_compatible_versions(transformers_version, autoawq_version):
            print("You have compatible versions of transformers and autoawq installed.")
            print("Would you like to proceed with the installation anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Installation aborted.")
                return
        else:
            print("⚠️ Incompatible versions detected. Will update packages.")
    
    # First uninstall potentially conflicting packages
    print("Uninstalling existing autoawq to prevent conflicts...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "autoawq"
        ])
    except:
        print("No existing autoawq to uninstall or uninstall failed - continuing...")
    
    # Install the latest transformers separately
    print("Installing transformers 4.49.0...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "transformers==4.49.0", "--upgrade"
    ])
    
    # Install autoawq with compatible version
    print("Installing autoawq 0.2.8 with --no-deps to prevent transformers downgrade...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "autoawq==0.2.8", "--no-deps"
    ])
    
    # Install other dependencies from requirements
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    # Parse requirements, skipping transformers and autoawq which we've handled
    with open(requirements_path, "r") as f:
        requirements = f.read()
    
    # Extract valid requirement lines
    requirements_list = []
    for line in requirements.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("--"):
            # Skip transformers and autoawq lines as we've handled them
            if not any(pkg in line for pkg in ["transformers", "autoawq"]):
                requirements_list.append(line)
    
    # Install other dependencies
    for req in requirements_list:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", req
            ])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {req}: {e}")
    
    # Install qwen_vl_utils
    try:
        print("Downloading qwen_vl_utils.py...")
        import urllib.request
        url = "https://raw.githubusercontent.com/QwenLM/Qwen-VL/main/qwen_vl_utils.py"
        urllib.request.urlretrieve(
            url, 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_vl_utils.py")
        )
        print("Successfully downloaded qwen_vl_utils.py")
    except Exception as e:
        print(f"Warning: Failed to download qwen_vl_utils.py: {e}")
    
    # Verify final versions
    print("\nVerifying installed package versions:")
    new_transformers_version = get_installed_version("transformers")
    new_autoawq_version = get_installed_version("autoawq")
    print(f"transformers: {new_transformers_version}")
    print(f"autoawq: {new_autoawq_version}")
    
    if is_compatible_versions(new_transformers_version, new_autoawq_version):
        print("✅ Installation successful! You have compatible versions installed.")
    else:
        print("⚠️ Warning: The installed versions may not be fully compatible.")
        print("You need transformers 4.49.0 or higher and autoawq 0.2.8.")
    
    print("\nNote: If you're using ComfyUI in a virtual environment, make sure to run this script in the same environment.")
    print("You may need to restart ComfyUI for changes to take effect.")

if __name__ == "__main__":
    install_dependencies() 
