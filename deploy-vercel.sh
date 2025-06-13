#!/bin/bash

# Golf Prediction App - Vercel Deployment Script
# Automates the deployment process to Vercel

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Golf Prediction → Vercel${NC}"
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

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    print_success "Git is available"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a Git repository. Please initialize Git first:"
        echo "  git init"
        echo "  git add ."
        echo "  git commit -m 'Initial commit'"
        exit 1
    fi
    print_success "Git repository detected"
    
    # Check if Vercel files exist
    if [ ! -f "vercel.json" ]; then
        print_error "vercel.json not found. Please ensure Vercel configuration files are present."
        exit 1
    fi
    print_success "Vercel configuration found"
    
    if [ ! -f "vercel_app.py" ]; then
        print_error "vercel_app.py not found. Please ensure Vercel app file is present."
        exit 1
    fi
    print_success "Vercel app file found"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. You'll need to set environment variables in Vercel dashboard."
    else
        print_success "Environment file found"
    fi
}

# Check if Vercel CLI is installed
check_vercel_cli() {
    if command -v vercel &> /dev/null; then
        print_success "Vercel CLI is installed"
        return 0
    else
        print_warning "Vercel CLI not found"
        return 1
    fi
}

# Install Vercel CLI
install_vercel_cli() {
    print_info "Installing Vercel CLI..."
    
    if command -v npm &> /dev/null; then
        npm install -g vercel
        print_success "Vercel CLI installed"
    elif command -v yarn &> /dev/null; then
        yarn global add vercel
        print_success "Vercel CLI installed"
    else
        print_error "Neither npm nor yarn found. Please install Node.js first."
        print_info "Visit: https://nodejs.org/"
        exit 1
    fi
}

# Prepare for deployment
prepare_deployment() {
    print_info "Preparing for deployment..."
    
    # Ensure all changes are committed
    if ! git diff-index --quiet HEAD --; then
        print_warning "You have uncommitted changes. Committing them now..."
        git add .
        git commit -m "Prepare for Vercel deployment - $(date)"
        print_success "Changes committed"
    else
        print_success "All changes are committed"
    fi
    
    # Check if remote origin exists
    if ! git remote get-url origin &> /dev/null; then
        print_error "No remote origin found. Please add your GitHub repository:"
        echo "  git remote add origin https://github.com/yourusername/usopenprediction.git"
        echo "  git push -u origin main"
        exit 1
    fi
    print_success "Remote origin configured"
    
    # Push to GitHub
    print_info "Pushing to GitHub..."
    git push origin main
    print_success "Code pushed to GitHub"
}

# Deploy with Vercel CLI
deploy_with_cli() {
    print_info "Deploying with Vercel CLI..."
    
    # Login to Vercel
    print_info "Please login to Vercel..."
    vercel login
    
    # Deploy
    print_info "Starting deployment..."
    vercel --prod
    
    print_success "Deployment completed!"
}

# Show deployment instructions
show_manual_instructions() {
    print_info "Manual Deployment Instructions:"
    echo ""
    echo "1. Go to https://vercel.com/dashboard"
    echo "2. Click 'New Project'"
    echo "3. Import your GitHub repository"
    echo "4. Configure environment variables:"
    echo "   - DATAGOLF_API_KEY: your_api_key_here"
    echo "   - SECRET_KEY: golf-prediction-vercel-2025"
    echo "   - FLASK_ENV: production"
    echo "5. Click 'Deploy'"
    echo ""
    print_info "Your app will be live at: https://your-app-name.vercel.app"
}

# Main deployment function
deploy_to_vercel() {
    print_header
    
    # Check prerequisites
    check_prerequisites
    
    # Prepare deployment
    prepare_deployment
    
    # Check for Vercel CLI
    if check_vercel_cli; then
        # CLI is available, use it
        deploy_with_cli
    else
        # Ask user if they want to install CLI
        echo ""
        read -p "Vercel CLI not found. Install it now? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_vercel_cli
            deploy_with_cli
        else
            print_info "Skipping CLI installation."
            show_manual_instructions
        fi
    fi
    
    echo ""
    print_success "Deployment process completed!"
    echo ""
    print_info "Next steps:"
    echo "1. Check your Vercel dashboard for deployment status"
    echo "2. Test your application at the provided URL"
    echo "3. Set up custom domain if needed"
    echo "4. Monitor performance and logs"
    echo ""
    print_info "Useful commands:"
    echo "  vercel logs your-app-name    # View logs"
    echo "  vercel domains               # Manage domains"
    echo "  vercel env                   # Manage environment variables"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  deploy      Deploy to Vercel (default)"
    echo "  check       Check prerequisites only"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  $0 check"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy_to_vercel
        ;;
    check)
        print_header
        check_prerequisites
        print_success "All prerequisites met!"
        ;;
    help|*)
        show_usage
        ;;
esac
