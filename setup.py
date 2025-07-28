"""
Setup script for BERT Text Classification Project
This script helps set up the environment and run the project components
"""

import os
import subprocess
import sys
import zipfile
import gdown
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10.x detected - Compatible!")
        return True
    elif version.major == 3 and version.minor >= 8:
        print("⚠️  Python version is compatible but not 3.10.x")
        return True
    else:
        print("❌ Python version not compatible. Please use Python 3.8+")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'saved_model',
        'tokenizer', 
        'data',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_model_files():
    """Download pre-trained model files from Google Drive"""
    print("\n" + "="*60)
    print("MODEL FILES DOWNLOAD")
    print("="*60)
    
    # Note: These are placeholder URLs - you would need to upload your actual model files
    model_files = {
        'saved_model.zip': {
            'url': 'https://drive.google.com/uc?id=YOUR_SAVED_MODEL_FILE_ID',
            'path': './saved_model.zip',
            'extract_to': './saved_model/'
        },
        'tokenizer.zip': {
            'url': 'https://drive.google.com/uc?id=YOUR_TOKENIZER_FILE_ID', 
            'path': './tokenizer.zip',
            'extract_to': './tokenizer/'
        }
    }
    
    print("📋 Model files need to be downloaded from Google Drive:")
    print("\nAfter training the model, upload these files to Google Drive:")
    print("1. saved_model/ folder (compressed as saved_model.zip)")
    print("2. tokenizer/ folder (compressed as tokenizer.zip)")
    print("\nThen update the Google Drive URLs in this script.")
    
    # Check if model files already exist
    if os.path.exists('./saved_model/config.json') and os.path.exists('./tokenizer/tokenizer.json'):
        print("✅ Model files already exist locally!")
        return True
    
    print("\n⚠️  Model files not found locally.")
    print("Please run the training notebook first or download the pre-trained models.")
    return False

def run_training():
    """Run the training notebook"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    try:
        print("Starting Jupyter notebook for training...")
        print("Please run the bert_text_classifier.ipynb notebook")
        subprocess.call(["jupyter", "notebook", "bert_text_classifier.ipynb"])
    except FileNotFoundError:
        print("❌ Jupyter not found. Installing jupyter...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jupyter"])
        print("Now please run: jupyter notebook bert_text_classifier.ipynb")

def run_gui():
    """Launch the Streamlit GUI"""
    print("\n" + "="*60)
    print("LAUNCHING GUI")
    print("="*60)
    
    try:
        print("Starting Streamlit GUI...")
        subprocess.call(["streamlit", "run", "app.py"])
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install requirements first.")

def run_inference_demo():
    """Run inference demo"""
    print("\n" + "="*60)
    print("RUNNING INFERENCE DEMO")
    print("="*60)
    
    try:
        subprocess.call([sys.executable, "inference.py"])
    except Exception as e:
        print(f"❌ Error running inference: {e}")

def main():
    """Main setup function"""
    print("🤖 BERT Text Classification Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n" + "="*50)
    print("SETUP OPTIONS")
    print("="*50)
    print("1. Train new model (run Jupyter notebook)")
    print("2. Download pre-trained model files")
    print("3. Launch GUI (requires trained model)")
    print("4. Run inference demo (requires trained model)")
    print("5. Install requirements only")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                run_training()
                break
            elif choice == '2':
                download_model_files()
                break
            elif choice == '3':
                if os.path.exists('./saved_model/config.json'):
                    run_gui()
                else:
                    print("❌ Model not found. Please train the model first (option 1)")
                break
            elif choice == '4':
                if os.path.exists('./saved_model/config.json'):
                    run_inference_demo()
                else:
                    print("❌ Model not found. Please train the model first (option 1)")
                break
            elif choice == '5':
                print("✅ Requirements already installed!")
                break
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-6.")
        
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled.")
            break

if __name__ == "__main__":
    main()