import os
import subprocess
import sys
from datetime import datetime
import json

# Use modern importlib.metadata instead of deprecated pkg_resources
try:
    from importlib.metadata import distributions
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import distributions

def get_installed_packages():
    """Get installed packages using modern importlib.metadata"""
    try:
        return {dist.metadata['name']: dist.version for dist in distributions()}
    except Exception as e:
        print(f"Warning: Could not get package info: {e}")
        return {}

def install_latest_and_record_versions(method="requirements"):
    """
    Install latest versions and record them using specified method
    
    Args:
        method (str): 'requirements', 'conda', 'mlflow-only', or 'all'
    """
    # Updated package list based on XGBoost as main model
    base_packages = [
        "mediapipe", 
        "opencv-python", 
        "numpy", 
        "xgboost",           # Main model
        "scikit-learn",      # For metrics and data splitting
        "mlflow", 
        "matplotlib", 
        "seaborn", 
        "pandas"
    ]
    
    print("🚀 Installing latest versions of required packages...")
    print("This may take several minutes...\n")
    
    # Install packages
    failed_packages = []
    for i, package in enumerate(base_packages, 1):
        print(f"[{i}/{len(base_packages)}] Installing/upgrading {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error with {package}: {e}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠ Warning: Failed to install: {', '.join(failed_packages)}")
    
    # Record versions based on method
    print(f"\n📝 Recording package versions using method: {method}")
    
    if method in ["requirements", "all"]:
        generate_requirements_txt()
    
    if method in ["conda", "all"]:
        generate_conda_environment()
    
    if method in ["mlflow-only", "all"]:
        save_environment_snapshot()

def generate_requirements_txt():
    """Generate organized requirements.txt"""
    try:
        # Get all installed packages
        result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        all_packages = result.decode().split('\n')
        
        # Filter for relevant packages (updated for XGBoost)
        relevant_keywords = [
            'mediapipe', 'opencv', 'numpy', 'xgboost', 'scikit', 'sklearn', 
            'mlflow', 'matplotlib', 'seaborn', 'pandas', 'pillow', 'protobuf', 
            'scipy', 'joblib', 'threadpoolctl', 'click', 'flask', 'jinja2'
        ]
        
        filtered_packages = []
        for line in all_packages:
            if line.strip() and '==' in line:
                package_name = line.split('==')[0].lower()
                if any(keyword in package_name for keyword in relevant_keywords):
                    filtered_packages.append(line.strip())
        
        # Write organized requirements.txt
        with open('requirements.txt', 'w') as f:
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Hand Gesture Recognition Project Requirements\n")
            f.write("# Main model: XGBoost\n")
            f.write("# Auto-generated with latest package versions\n\n")
            
            # Categorize packages (updated for XGBoost)
            categories = {
                'Computer Vision': ['mediapipe', 'opencv', 'pillow'],
                'Machine Learning': ['xgboost', 'scikit', 'sklearn', 'mlflow'],
                'Data Science': ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy'],
                'Dependencies': []
            }
            
            categorized = {cat: [] for cat in categories}
            
            for package in sorted(filtered_packages):
                pkg_name = package.split('==')[0].lower()
                categorized_flag = False
                
                for category, keywords in categories.items():
                    if category != 'Dependencies' and any(kw in pkg_name for kw in keywords):
                        categorized[category].append(package)
                        categorized_flag = True
                        break
                
                if not categorized_flag:
                    categorized['Dependencies'].append(package)
            
            # Write categorized packages
            for category, packages in categorized.items():
                if packages:
                    f.write(f"# {category}\n")
                    for pkg in packages:
                        f.write(f"{pkg}\n")
                    f.write("\n")
        
        print(f"✓ requirements.txt generated with {len(filtered_packages)} packages")
        
    except Exception as e:
        print(f"Error generating requirements.txt: {e}")

def generate_conda_environment():
    """Generate conda environment.yml if in conda environment"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env and conda_env != 'base':
            result = subprocess.check_output(["conda", "env", "export", "--no-builds"]).decode()
            with open('environment.yml', 'w') as f:
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(result)
            print("✓ environment.yml generated")
        else:
            print("ℹ Not in conda environment, skipping environment.yml")
    except Exception as e:
        print(f"ℹ Conda not available: {e}")

def save_environment_snapshot():
    """Save environment snapshot for MLflow tracking"""
    try:
        installed_packages = get_installed_packages()
        
        env_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "main_model": "XGBoost",
            "packages": installed_packages
        }
        
        os.makedirs("environment_snapshots", exist_ok=True)
        snapshot_file = f"environment_snapshots/env_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(env_snapshot, f, indent=2)
        
        print(f"✓ Environment snapshot saved to {snapshot_file}")
        
    except Exception as e:
        print(f"Error saving environment snapshot: {e}")

def setup_environment():
    """Setup the project environment"""
    print("\n" + "="*60)
    print("🏗️  Setting up Hand Gesture Recognition environment...")
    print("Main Model: XGBoost")
    print("="*60)
    
    # Create necessary directories
    directories = ["Models", "mlflow_data", "data", "logs", "environment_snapshots"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")
    
    # Check for MediaPipe model
    landmarker_path = os.path.join("Models", "hand_landmarker.task")
    if os.path.exists(landmarker_path):
        print(f"✓ Found hand_landmarker.task in Models folder")
    else:
        print(f"⚠ Warning: hand_landmarker.task not found in Models folder")
        print("  Download from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
    
    # Create .gitignore
    create_gitignore()
    
    print("\n✓ Environment setup completed!")

def create_gitignore():
    """Create comprehensive .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/

# MLflow
mlflow_data/
mlruns/

# Models (except hand_landmarker.task)
Models/*.pkl
Models/*.json
Models/*.png
Models/*.h5
Models/*.pb
Models/*.model  # XGBoost model files
Models/*.bst    # XGBoost model files
!Models/hand_landmarker.task
!Models/.gitkeep

# Data
data/
*.csv
datasets/

# Logs
logs/
*.log

# Environment snapshots
environment_snapshots/

# Jupyter Notebooks
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Images (except sample images)
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.tiff
!sample_images/
"""
    
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore file")

def check_installation():
    """Check if all required packages are installed correctly"""
    print("\n" + "="*50)
    print("🔍 Checking package installations...")
    print("="*50)
    
    # Updated package list with XGBoost
    packages_to_check = [
        ("mediapipe", "mediapipe"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("xgboost", "xgboost"),        # Main model
        ("scikit-learn", "sklearn"),
        ("mlflow", "mlflow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas")
    ]
    
    working_packages = []
    failed_packages = []
    
    for package_name, import_name in packages_to_check:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            
            # Special handling for XGBoost to show it's the main model
            if package_name == "xgboost":
                print(f"✓ {package_name} (v{version}) [MAIN MODEL]")
            else:
                print(f"✓ {package_name} (v{version})")
            
            working_packages.append(package_name)
        except ImportError:
            print(f"✗ {package_name} - Not installed or not working")
            failed_packages.append(package_name)
    
    print(f"\n📊 Summary: {len(working_packages)}/{len(packages_to_check)} packages working")
    
    if failed_packages:
        print(f"❌ Failed packages: {', '.join(failed_packages)}")
        return False
    else:
        print("✅ All packages are working correctly!")
        return True

def show_next_steps():
    """Show next steps after setup"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! Next steps:")
    print("="*60)
    print("1. 📁 Organize your dataset:")
    print("   data/gestures/")
    print("   ├── gesture1/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   └── gesture2/")
    print("       └── ...")
    print()
    print("2. 🏋️  Train your XGBoost model:")
    print("   python train_model.py --dataset data/gestures")
    print()
    print("3. 🧪 Run multiple experiments:")
    print("   python run_experiments.py data/gestures")
    print()
    print("4. 📊 View results in MLflow:")
    print("   mlflow ui --backend-store-uri file://./mlflow_data")
    print()
    print("5. 🔮 Test model inference:")
    print("   python inference.py --model Models/[model_file] --classes Models/[class_file] --webcam")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Hand Gesture Recognition Environment (XGBoost)')
    parser.add_argument('--install-latest', action='store_true', 
                       help='Install latest versions and generate requirements.txt')
    parser.add_argument('--method', choices=['requirements', 'conda', 'mlflow-only', 'all'], 
                       default='requirements',
                       help='Method to record package versions')
    parser.add_argument('--check', action='store_true', 
                       help='Check if packages are installed correctly')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only create directories and setup project structure')
    
    args = parser.parse_args()
    
    if args.install_latest:
        install_latest_and_record_versions(args.method)
        setup_environment()
        if check_installation():
            show_next_steps()
    elif args.check:
        check_installation()
    elif args.setup_only:
        setup_environment()
    else:
        print("Use --install-latest to install packages and setup environment")
        print("Use --check to verify installations")
        print("Use --setup-only to just create project structure")