# 🔧 Vercel Environment Variables Setup

## Quick Setup Guide

### 1. Access Vercel Dashboard
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your `golf-prediction-app` project
3. Click on **Settings** tab
4. Click on **Environment Variables** in the sidebar

### 2. Add Environment Variables

Add these environment variables for enhanced functionality:

#### Required Variables:
```
DATAGOLF_API_KEY = be1e0f4c0d741ab978b3fded7e8c
```

#### Optional Variables:
```
USE_REAL_DATA = true
DATABASE_PATH = data/golf_predictions.db
```

### 3. Environment Variable Details

| Variable | Value | Purpose |
|----------|-------|---------|
| `DATAGOLF_API_KEY` | Your API key | Access DataGolf API for live data |
| `USE_REAL_DATA` | `true` or `false` | Switch between real and mock data |
| `DATABASE_PATH` | `data/golf_predictions.db` | Path to SQLite database |

### 4. Setting Variables in Vercel

For each variable:
1. Click **Add New**
2. Enter **Name** (e.g., `DATAGOLF_API_KEY`)
3. Enter **Value** (e.g., `be1e0f4c0d741ab978b3fded7e8c`)
4. Select **Environment**: `Production`
5. Click **Save**

### 5. Redeploy After Changes

After adding environment variables:
1. Go to **Deployments** tab
2. Click **Redeploy** on the latest deployment
3. Wait for deployment to complete

## 🧪 Testing Configuration

### Check Configuration Status
Visit: `https://your-app.vercel.app/api/config`

This endpoint shows:
- Environment variable status
- Database availability
- DataGolf API configuration
- Current data source

### Health Check
Visit: `https://your-app.vercel.app/api/health`

Shows system status and configuration.

## 📊 Data Sources

### Mock Data (Default)
- Uses sample predictions
- Always available
- Good for testing

### Real Data (When Enabled)
- Fetches from SQLite database
- Falls back to DataGolf API
- Falls back to mock data if unavailable

## 🔄 Switching Data Sources

### To Use Real Data:
1. Set `USE_REAL_DATA = true`
2. Ensure database is available OR
3. Configure DataGolf API key
4. Redeploy

### To Use Mock Data:
1. Set `USE_REAL_DATA = false` (or remove variable)
2. Redeploy

## 🎯 Next Steps

1. **Set environment variables** in Vercel dashboard
2. **Redeploy** the application
3. **Test endpoints** to verify configuration
4. **Upload database** if you have real prediction data
5. **Monitor performance** through Vercel analytics

Your app will automatically detect and use the best available data source!
