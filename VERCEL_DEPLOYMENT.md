# 🚀 Golf Prediction App - Vercel Deployment Guide

Deploy your Golf Prediction application to Vercel in minutes!

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
3. **DataGolf API Key** - Your API key for data access

## 🔧 Pre-Deployment Setup

### 1. Prepare Your Repository

Make sure your repository has these Vercel-specific files:
- ✅ `vercel.json` - Vercel configuration
- ✅ `vercel_app.py` - Serverless-optimized Flask app
- ✅ `requirements-vercel.txt` - Lightweight dependencies

### 2. Push to GitHub

```bash
git add .
git commit -m "Add Vercel deployment configuration"
git push origin main
```

## 🚀 Deployment Steps

### Method 1: Vercel Dashboard (Recommended)

1. **Go to Vercel Dashboard**
   - Visit [vercel.com/dashboard](https://vercel.com/dashboard)
   - Sign in with GitHub

2. **Import Project**
   - Click "New Project"
   - Select your `usopenprediction` repository
   - Click "Import"

3. **Configure Environment Variables**
   - In the deployment settings, add:
   ```
   DATAGOLF_API_KEY = your_api_key_here
   SECRET_KEY = your_secret_key_here
   FLASK_ENV = production
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete (2-3 minutes)
   - Your app will be live at `https://your-app-name.vercel.app`

### Method 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy from Terminal**
   ```bash
   cd /path/to/usopenprediction
   vercel
   ```

4. **Set Environment Variables**
   ```bash
   vercel env add DATAGOLF_API_KEY
   vercel env add SECRET_KEY
   vercel env add FLASK_ENV
   ```

5. **Redeploy with Environment Variables**
   ```bash
   vercel --prod
   ```

## 🔐 Environment Variables Setup

In Vercel Dashboard → Project Settings → Environment Variables, add:

| Variable | Value | Environment |
|----------|-------|-------------|
| `DATAGOLF_API_KEY` | `your_api_key_here` | Production |
| `SECRET_KEY` | `golf-prediction-vercel-2025` | Production |
| `FLASK_ENV` | `production` | Production |

## 📱 Application URLs

After deployment, your app will be available at:

- **Main Dashboard**: `https://your-app.vercel.app/`
- **Predictions**: `https://your-app.vercel.app/predictions`
- **Value Picks**: `https://your-app.vercel.app/value-picks`
- **Analytics**: `https://your-app.vercel.app/analytics`
- **Model Evaluation**: `https://your-app.vercel.app/evaluation`
- **API Health**: `https://your-app.vercel.app/api/health`

## 🔧 Vercel Configuration Details

### vercel.json Configuration
```json
{
  "version": 2,
  "name": "golf-prediction-app",
  "builds": [
    {
      "src": "vercel_app.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "vercel_app.py"
    }
  ],
  "env": {
    "FLASK_ENV": "production"
  },
  "functions": {
    "vercel_app.py": {
      "maxDuration": 30
    }
  }
}
```

### Key Features:
- **Serverless Functions**: Each request runs in an isolated function
- **Auto-scaling**: Handles traffic spikes automatically
- **Global CDN**: Fast loading worldwide
- **HTTPS**: Automatic SSL certificates
- **Custom Domains**: Add your own domain

## 🎯 Serverless Optimizations

The Vercel deployment includes several optimizations:

1. **Lightweight Dependencies**: Uses `requirements-vercel.txt`
2. **Error Handling**: Graceful fallbacks for missing data
3. **Mock Data**: Demo data when database is unavailable
4. **Fast Cold Starts**: Optimized imports and initialization

## 🔍 Monitoring & Debugging

### Check Deployment Status
- Vercel Dashboard → Project → Functions tab
- View real-time logs and metrics

### Debug Issues
1. **Check Function Logs**
   ```bash
   vercel logs your-app-name
   ```

2. **Test API Endpoints**
   ```bash
   curl https://your-app.vercel.app/api/health
   ```

3. **Monitor Performance**
   - Vercel Analytics (available in dashboard)
   - Function execution time and memory usage

## 🚨 Common Issues & Solutions

### Issue: Import Errors
**Solution**: Check `requirements-vercel.txt` has all needed packages

### Issue: Function Timeout
**Solution**: Optimize database queries or increase `maxDuration`

### Issue: Memory Limit
**Solution**: Reduce data processing or use external database

### Issue: Cold Start Delays
**Solution**: Implement warming functions or use Vercel Pro

## 🔄 Continuous Deployment

Vercel automatically redeploys when you push to GitHub:

1. **Make Changes**
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```

2. **Automatic Deployment**
   - Vercel detects the push
   - Builds and deploys automatically
   - Updates live site in 2-3 minutes

## 📊 Performance Optimization

### For Better Performance:
1. **Use Vercel Pro** for faster builds and more resources
2. **Implement Caching** for API responses
3. **Optimize Images** using Vercel Image Optimization
4. **Use Edge Functions** for faster global response

### Database Considerations:
- **SQLite**: Works but limited in serverless
- **PostgreSQL**: Better for production (use Vercel Postgres)
- **Redis**: For caching (use Vercel KV)

## 🎉 Success Checklist

After deployment, verify:
- [ ] App loads at Vercel URL
- [ ] All pages render correctly
- [ ] API endpoints respond
- [ ] Environment variables are set
- [ ] No console errors
- [ ] Mobile responsiveness works

## 🆘 Support

If you encounter issues:

1. **Check Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
2. **View Function Logs**: Vercel Dashboard → Functions
3. **Test Locally**: `python vercel_app.py`
4. **Check API Health**: `/api/health` endpoint

## 🎯 Next Steps

After successful deployment:
1. **Custom Domain**: Add your own domain in Vercel settings
2. **Analytics**: Enable Vercel Analytics for insights
3. **Monitoring**: Set up alerts for errors
4. **Performance**: Monitor and optimize based on usage

Your Golf Prediction app is now live on Vercel! 🏌️‍♂️⛳
