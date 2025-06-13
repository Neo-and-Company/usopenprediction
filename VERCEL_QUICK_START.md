# 🚀 Quick Vercel Deployment Guide

Deploy your Golf Prediction app to Vercel in 3 simple steps!

## 🎯 One-Click Deployment

### Option 1: Automated Script (Recommended)

```bash
./deploy-vercel-simple.sh
```

This script will:
- ✅ Check prerequisites
- ✅ Initialize Git (if needed)
- ✅ Push to GitHub
- ✅ Install Vercel CLI (if needed)
- ✅ Deploy to Vercel
- ✅ Provide next steps

### Option 2: Manual Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Vercel"
   git push origin main
   ```

2. **Deploy with Vercel**
   ```bash
   npm install -g vercel
   vercel login
   vercel --prod
   ```

3. **Set Environment Variables**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Select your project → Settings → Environment Variables
   - Add:
     - `DATAGOLF_API_KEY`: `be1e0f4c0d741ab978b3fded7e8c`
     - `SECRET_KEY`: `golf-prediction-vercel-2025`
     - `FLASK_ENV`: `production`

## 🔧 What's Included

- ✅ **Self-contained app** (`vercel_app.py`) - works without complex dependencies
- ✅ **Mock data fallback** - app works even without database
- ✅ **Optimized configuration** (`vercel.json`) - fast builds and deployments
- ✅ **Lightweight requirements** (`requirements-vercel.txt`) - minimal dependencies
- ✅ **Error handling** - graceful fallbacks for missing data

## 📱 Your App URLs

After deployment, your app will be available at:

- **Dashboard**: `https://your-app.vercel.app/`
- **Predictions**: `https://your-app.vercel.app/predictions`
- **Value Picks**: `https://your-app.vercel.app/value-picks`
- **Analytics**: `https://your-app.vercel.app/analytics`
- **Model Evaluation**: `https://your-app.vercel.app/evaluation`
- **Health Check**: `https://your-app.vercel.app/api/health`

## 🎉 Success Checklist

- [ ] App loads without errors
- [ ] All pages render correctly
- [ ] API endpoints respond
- [ ] Environment variables are set
- [ ] Health check returns "healthy"

## 🆘 Troubleshooting

### Common Issues:

1. **Build fails**: Check `requirements-vercel.txt` has all dependencies
2. **500 errors**: Check Vercel function logs in dashboard
3. **Environment variables**: Verify they're set in Vercel dashboard
4. **Import errors**: The app uses self-contained code to avoid this

### Quick Fixes:

```bash
# Test locally first
python vercel_app.py

# Check health endpoint
curl https://your-app.vercel.app/api/health

# View Vercel logs
vercel logs your-app-name
```

## 🎯 Next Steps

1. **Custom Domain**: Add your domain in Vercel settings
2. **Analytics**: Enable Vercel Analytics
3. **Monitoring**: Set up error alerts
4. **Database**: Consider upgrading to PostgreSQL for production

Your Golf Prediction app is now live! 🏌️‍♂️⛳
