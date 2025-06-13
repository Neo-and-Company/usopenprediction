#!/bin/bash

# Golf Prediction Application Deployment Script
# Supports multiple deployment methods

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Golf Prediction Deployment${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your API keys before continuing"
            exit 1
        else
            print_error ".env.example not found. Please create .env file manually"
            exit 1
        fi
    fi
    print_success "Environment file found"
}

# Install dependencies
install_deps() {
    print_header
    echo "Installing dependencies..."
    
    if command -v pip &> /dev/null; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "pip not found. Please install Python and pip first"
        exit 1
    fi
}

# Local development deployment
deploy_local() {
    print_header
    echo "Deploying for local development..."
    
    check_env
    install_deps
    
    echo "Starting Flask development server..."
    export FLASK_ENV=development
    python app.py
}

# Production deployment with Gunicorn
deploy_production() {
    print_header
    echo "Deploying for production..."
    
    check_env
    install_deps
    
    # Create necessary directories
    mkdir -p data logs static/css static/js templates
    
    echo "Starting production server with Gunicorn..."
    export FLASK_ENV=production
    gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 120 --access-logfile logs/access.log --error-logfile logs/error.log wsgi:application
}

# Docker deployment
deploy_docker() {
    print_header
    echo "Deploying with Docker..."
    
    check_env
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first"
        exit 1
    fi
    
    echo "Building Docker image..."
    docker build -t golf-prediction .
    print_success "Docker image built"
    
    echo "Starting Docker container..."
    docker run -d \
        --name golf-prediction-app \
        -p 8080:8080 \
        --env-file .env \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        golf-prediction
    
    print_success "Docker container started"
    echo "Application available at: http://localhost:8080"
}

# Docker Compose deployment
deploy_docker_compose() {
    print_header
    echo "Deploying with Docker Compose..."
    
    check_env
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose first"
        exit 1
    fi
    
    echo "Starting services with Docker Compose..."
    docker-compose up -d --build
    
    print_success "Services started with Docker Compose"
    echo "Application available at: http://localhost:8080"
}

# Heroku deployment
deploy_heroku() {
    print_header
    echo "Preparing for Heroku deployment..."
    
    if ! command -v heroku &> /dev/null; then
        print_error "Heroku CLI not found. Please install Heroku CLI first"
        exit 1
    fi
    
    # Create Procfile if it doesn't exist
    if [ ! -f Procfile ]; then
        echo "web: gunicorn wsgi:application" > Procfile
        print_success "Procfile created"
    fi
    
    # Create runtime.txt if it doesn't exist
    if [ ! -f runtime.txt ]; then
        echo "python-3.11.0" > runtime.txt
        print_success "runtime.txt created"
    fi
    
    echo "To complete Heroku deployment:"
    echo "1. heroku create your-app-name"
    echo "2. heroku config:set DATAGOLF_API_KEY=your_api_key"
    echo "3. git add ."
    echo "4. git commit -m 'Deploy to Heroku'"
    echo "5. git push heroku main"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  local       Deploy for local development"
    echo "  production  Deploy for production with Gunicorn"
    echo "  docker      Deploy with Docker"
    echo "  compose     Deploy with Docker Compose"
    echo "  heroku      Prepare for Heroku deployment"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local"
    echo "  $0 production"
    echo "  $0 docker"
}

# Main script logic
case "${1:-help}" in
    local)
        deploy_local
        ;;
    production)
        deploy_production
        ;;
    docker)
        deploy_docker
        ;;
    compose)
        deploy_docker_compose
        ;;
    heroku)
        deploy_heroku
        ;;
    help|*)
        show_usage
        ;;
esac
