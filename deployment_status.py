#!/usr/bin/env python3
"""
Golf Prediction Application - Deployment Status Checker
Shows current deployment status and available options.
"""

import os
import sys
import subprocess
import requests
from datetime import datetime

def check_command(command):
    """Check if a command is available."""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_port(port):
    """Check if a port is in use."""
    try:
        response = requests.get(f'http://localhost:{port}/api/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def print_header():
    """Print application header."""
    print("=" * 60)
    print("🏌️  GOLF PREDICTION APPLICATION - DEPLOYMENT STATUS")
    print("=" * 60)
    print(f"📅 Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_environment():
    """Check environment setup."""
    print("🔧 ENVIRONMENT CHECK:")
    print("-" * 30)
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"✓ Python: {python_version}")
    
    # Required files
    required_files = ['.env', 'requirements.txt', 'app.py', 'wsgi.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}: Found")
        else:
            print(f"✗ {file}: Missing")
    
    # Environment variables
    api_key = os.environ.get('DATAGOLF_API_KEY')
    if api_key:
        print(f"✓ DATAGOLF_API_KEY: Set (***{api_key[-4:]})")
    else:
        print("⚠ DATAGOLF_API_KEY: Not set")
    
    print()

def check_dependencies():
    """Check installed dependencies."""
    print("📦 DEPENDENCIES CHECK:")
    print("-" * 30)
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'gunicorn', 'waitress', 'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}: Installed")
        except ImportError:
            print(f"✗ {package}: Missing")
    
    print()

def check_deployment_tools():
    """Check available deployment tools."""
    print("🛠️  DEPLOYMENT TOOLS:")
    print("-" * 30)
    
    tools = {
        'docker': 'Docker',
        'docker-compose': 'Docker Compose',
        'heroku': 'Heroku CLI',
        'git': 'Git'
    }
    
    for command, name in tools.items():
        if check_command(command):
            print(f"✓ {name}: Available")
        else:
            print(f"✗ {name}: Not installed")
    
    print()

def check_running_services():
    """Check currently running services."""
    print("🚀 RUNNING SERVICES:")
    print("-" * 30)
    
    ports_to_check = {
        5001: 'Development Server',
        8080: 'Production Server (Gunicorn)',
        3000: 'Alternative Port',
        5000: 'Flask Default'
    }
    
    for port, description in ports_to_check.items():
        if check_port(port):
            print(f"✓ Port {port}: {description} - RUNNING")
        else:
            print(f"✗ Port {port}: {description} - Not running")
    
    print()

def show_deployment_options():
    """Show available deployment options."""
    print("📋 DEPLOYMENT OPTIONS:")
    print("-" * 30)
    
    options = [
        ("Local Development", "./deploy.sh local", "Quick testing and development"),
        ("Production Server", "./deploy.sh production", "Production-ready with Gunicorn"),
        ("Docker Container", "./deploy.sh docker", "Containerized deployment"),
        ("Docker Compose", "./deploy.sh compose", "Full stack with services"),
        ("Heroku Preparation", "./deploy.sh heroku", "Cloud deployment setup")
    ]
    
    for name, command, description in options:
        print(f"🔹 {name}")
        print(f"   Command: {command}")
        print(f"   Description: {description}")
        print()

def show_quick_commands():
    """Show quick deployment commands."""
    print("⚡ QUICK COMMANDS:")
    print("-" * 30)
    print("# Start development server:")
    print("python app.py")
    print()
    print("# Start production server:")
    print("gunicorn --bind 0.0.0.0:8080 --workers 4 wsgi:application")
    print()
    print("# Build and run Docker:")
    print("docker build -t golf-prediction .")
    print("docker run -p 8080:8080 --env-file .env golf-prediction")
    print()
    print("# Docker Compose:")
    print("docker-compose up -d --build")
    print()

def show_urls():
    """Show application URLs."""
    print("🌐 APPLICATION URLS:")
    print("-" * 30)
    
    base_urls = ['http://localhost:5001', 'http://localhost:8080']
    endpoints = [
        ('Dashboard', '/'),
        ('Predictions', '/predictions'),
        ('Value Picks', '/value-picks'),
        ('Analytics', '/analytics'),
        ('Model Evaluation', '/evaluation'),
        ('API Health', '/api/health'),
        ('API Predictions', '/api/predictions')
    ]
    
    for base_url in base_urls:
        port = base_url.split(':')[-1]
        if check_port(int(port)):
            print(f"🟢 Server running on {base_url}")
            for name, endpoint in endpoints:
                print(f"   {name}: {base_url}{endpoint}")
            print()
            break
    else:
        print("🔴 No servers currently running")
        print("   Start a server to access the application")
        print()

def main():
    """Main function."""
    print_header()
    check_environment()
    check_dependencies()
    check_deployment_tools()
    check_running_services()
    show_deployment_options()
    show_quick_commands()
    show_urls()
    
    print("=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. Choose a deployment method from the options above")
    print("2. Run the corresponding command")
    print("3. Access the application at the provided URL")
    print("4. Check /api/health for system status")
    print("=" * 60)

if __name__ == '__main__':
    main()
