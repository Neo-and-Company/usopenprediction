# Golf Prediction Application - Deployment Guide

This guide covers multiple deployment options for the Golf Prediction application.

## 🚀 Quick Deployment Options

### Option 1: Local Development (Fastest)
```bash
./deploy.sh local
```
- Runs on `http://localhost:5001`
- Development mode with auto-reload
- Best for testing and development

### Option 2: Production Server
```bash
./deploy.sh production
```
- Runs on `http://localhost:8080`
- Production-ready with Gunicorn
- Optimized for performance

### Option 3: Docker (Recommended)
```bash
./deploy.sh docker
```
- Containerized deployment
- Consistent environment
- Easy to scale and manage

### Option 4: Docker Compose (Full Stack)
```bash
./deploy.sh compose
```
- Multi-container setup
- Includes optional Redis and PostgreSQL
- Production-ready with monitoring

## 📋 Prerequisites

### Required
- Python 3.11+
- DataGolf API key
- Git

### Optional (depending on deployment method)
- Docker & Docker Compose
- Heroku CLI
- Cloud provider account (AWS, GCP, Azure)

## 🔧 Setup Instructions

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd usopenprediction
cp .env.example .env
# Edit .env with your API key
```

### 2. Choose Deployment Method

#### Local Development
```bash
pip install -r requirements.txt
python app.py
```

#### Production with Gunicorn
```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:8080 --workers 4 wsgi:application
```

#### Docker
```bash
docker build -t golf-prediction .
docker run -p 8080:8080 --env-file .env golf-prediction
```

#### Docker Compose
```bash
docker-compose up -d --build
```

## ☁️ Cloud Deployment

### Heroku
```bash
# Install Heroku CLI first
./deploy.sh heroku
heroku create your-app-name
heroku config:set DATAGOLF_API_KEY=your_api_key
git push heroku main
```

### AWS EC2
1. Launch EC2 instance (Ubuntu 20.04+)
2. Install Docker
3. Clone repository
4. Run with Docker Compose

### Google Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/golf-prediction
gcloud run deploy --image gcr.io/PROJECT_ID/golf-prediction --platform managed
```

### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image golf-prediction .
az container create --resource-group myResourceGroup --name golf-prediction --image myregistry.azurecr.io/golf-prediction:latest
```

## 🔒 Environment Variables

Required environment variables:
```bash
DATAGOLF_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
FLASK_ENV=production
```

Optional environment variables:
```bash
DATABASE_PATH=data/golf_predictions.db
LOG_LEVEL=INFO
CORS_ORIGINS=*
REDIS_URL=redis://localhost:6379
```

## 📊 Monitoring and Logs

### Application Logs
- Development: Console output
- Production: `logs/golf_prediction.log`
- Docker: `docker logs container_name`

### Health Check
- Endpoint: `/api/health`
- Returns application status and database connectivity

### Performance Monitoring
- Built-in Flask metrics
- Custom evaluation metrics at `/evaluation`
- API response times logged

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8080
# Kill process
kill -9 PID
```

#### Database Issues
```bash
# Reset database
rm data/golf_predictions.db
python -c "from src.data_pipeline.database_manager import GolfPredictionDB; db = GolfPredictionDB(); db.create_schema()"
```

#### Docker Issues
```bash
# Clean up Docker
docker system prune -a
docker-compose down -v
```

### Performance Optimization

#### For High Traffic
- Increase Gunicorn workers: `--workers 8`
- Add Redis caching
- Use PostgreSQL instead of SQLite
- Enable gzip compression

#### For Large Datasets
- Implement database connection pooling
- Add pagination to API endpoints
- Use background tasks for data processing

## 📈 Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple instances
- Implement session storage (Redis)

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Enable caching layers

## 🔐 Security

### Production Security Checklist
- [ ] Use HTTPS (SSL/TLS certificates)
- [ ] Set secure environment variables
- [ ] Enable CORS restrictions
- [ ] Implement rate limiting
- [ ] Regular security updates
- [ ] Database access controls
- [ ] API key rotation

### Firewall Rules
```bash
# Allow HTTP/HTTPS traffic
ufw allow 80
ufw allow 443
ufw allow 8080
```

## 📞 Support

For deployment issues:
1. Check logs first
2. Verify environment variables
3. Test API endpoints
4. Check database connectivity

Application URLs after deployment:
- Dashboard: `http://your-domain/`
- API Health: `http://your-domain/api/health`
- Model Evaluation: `http://your-domain/evaluation`
