#!/bin/bash

# 🚀 One-Click Vercel Deployment for Golf Prediction App
# This script automates the entire deployment process

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏌️ Golf Prediction App - One-Click Vercel Deployment${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Check if we're in the right directory
if [ ! -f "vercel_app.py" ]; then
    echo -e "${RED}❌ Error: vercel_app.py not found. Please run this script from the project root.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found vercel_app.py${NC}"

# Step 2: Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit for Vercel deployment"
fi

# Step 3: Check if remote origin exists
if ! git remote get-url origin &> /dev/null; then
    echo -e "${YELLOW}⚠️  No remote origin found.${NC}"
    echo -e "${BLUE}Please add your GitHub repository URL:${NC}"
    read -p "Enter your GitHub repository URL (e.g., https://github.com/username/usopenprediction.git): " repo_url
    git remote add origin "$repo_url"
fi

# Step 4: Commit any changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Committing changes...${NC}"
    git add .
    git commit -m "Prepare for Vercel deployment - $(date)"
fi

# Step 5: Push to GitHub
echo -e "${BLUE}📤 Pushing to GitHub...${NC}"
git push -u origin main

# Step 6: Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}⚠️  Vercel CLI not found. Installing...${NC}"
    if command -v npm &> /dev/null; then
        npm install -g vercel
    elif command -v yarn &> /dev/null; then
        yarn global add vercel
    else
        echo -e "${RED}❌ Neither npm nor yarn found. Please install Node.js first.${NC}"
        echo -e "${BLUE}Visit: https://nodejs.org/${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Vercel CLI is ready${NC}"

# Step 7: Login to Vercel (if not already logged in)
echo -e "${BLUE}🔐 Checking Vercel authentication...${NC}"
if ! vercel whoami &> /dev/null; then
    echo -e "${YELLOW}Please login to Vercel:${NC}"
    vercel login
fi

# Step 8: Deploy to Vercel
echo -e "${BLUE}🚀 Deploying to Vercel...${NC}"
vercel --prod

echo -e "${GREEN}🎉 Deployment completed!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "1. Set environment variables in Vercel dashboard:"
echo -e "   - DATAGOLF_API_KEY: your_api_key_here"
echo -e "   - SECRET_KEY: golf-prediction-vercel-2025"
echo -e "2. Your app should be live at the URL shown above"
echo -e "3. Test the /api/health endpoint to verify deployment"

echo -e "${GREEN}✨ Happy golfing! ⛳${NC}"
