# 🚀 Production Deployment Guide

Complete guide to deploy AI ML Studio to production.

---

## 📋 Pre-Deployment Checklist

### Infrastructure Requirements

- [ ] Linux server (Ubuntu 22.04 LTS recommended)
- [ ] NVIDIA GPU with CUDA 12.1+ (RTX 5090 or similar)
- [ ] 64GB+ RAM (128GB recommended)
- [ ] 500GB+ SSD storage
- [ ] Docker & Docker Compose installed
- [ ] Domain name configured
- [ ] SSL certificate obtained

### Services Requirements

- [ ] PostgreSQL 16+ (or managed service)
- [ ] Redis 7+ (or managed service)
- [ ] Object storage (S3/MinIO) for models
- [ ] CDN for static assets (optional)

---

## 🔧 Step 1: Server Setup

### 1.1 Update System

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git
```

### 1.2 Install NVIDIA Drivers

```bash
# Add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver (check latest version)
sudo apt install -y nvidia-driver-545

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### 1.3 Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 1.4 Install Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

---

## 📦 Step 2: Deploy Application

### 2.1 Clone Repository

```bash
cd /opt
sudo git clone https://github.com/your-org/ai-ml-studio.git
cd ai-ml-studio
```

### 2.2 Configure Environment

```bash
# Copy environment file
cp .env.example .env.production

# Edit configuration
sudo nano .env.production
```

**Production .env:**
```env
# Environment
ENVIRONMENT=production
DEBUG=false

# API Keys (use secrets manager in production)
ANTHROPIC_API_KEY=your_production_key
OPENAI_API_KEY=your_production_key
WANDB_API_KEY=your_production_key

# Database (use managed service)
DATABASE_URL=postgresql://user:pass@your-db-host:5432/aimlstudio

# Redis (use managed service)
REDIS_URL=redis://your-redis-host:6379/0

# Security
SECRET_KEY=generate_strong_random_key_here
JWT_ALGORITHM=HS256

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Storage
MODEL_STORAGE_PATH=/data/models
DATA_STORAGE_PATH=/data/datasets

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
```

### 2.3 Create Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: aiml-backend-prod
    restart: unless-stopped
    command: uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
    ports:
      - "8000:8000"
    env_file:
      - .env.production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '16'
          memory: 64G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - aiml-network

  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: aiml-celery-prod
    restart: unless-stopped
    command: celery -A backend.core.celery worker --loglevel=info --concurrency=8
    env_file:
      - .env.production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - backend
    networks:
      - aiml-network

  nginx:
    image: nginx:alpine
    container_name: aiml-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - backend
    networks:
      - aiml-network

volumes:
  models:
  data:
  logs:

networks:
  aiml-network:
    driver: bridge
```

### 2.4 Configure Nginx

Create `nginx/nginx.conf`:

```nginx
upstream backend {
    server backend:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    # Proxy Configuration
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check endpoint (no rate limit)
    location /health {
        proxy_pass http://backend/health;
        access_log off;
    }

    # Static files (if any)
    location /static {
        alias /var/www/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 2.5 Deploy

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f backend

# Verify health
curl https://api.yourdomain.com/health
```

---

## 🔒 Step 3: Security Hardening

### 3.1 Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow only from specific IPs (optional)
sudo ufw allow from YOUR_IP to any port 22

# Check status
sudo ufw status
```

### 3.2 Fail2Ban

```bash
# Install Fail2Ban
sudo apt install -y fail2ban

# Configure
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local

# Add custom rule for API
cat << EOF | sudo tee /etc/fail2ban/filter.d/aiml-api.conf
[Definition]
failregex = ^<HOST>.*"(GET|POST|PUT|DELETE).*" 40[134]
ignoreregex =
EOF

# Restart
sudo systemctl restart fail2ban
```

### 3.3 SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal (already configured)
sudo certbot renew --dry-run
```

---

## 📊 Step 4: Monitoring Setup

### 4.1 Configure Prometheus

Add to `docker-compose.prod.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: aiml-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - aiml-network

  grafana:
    image: grafana/grafana:latest
    container_name: aiml-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_SERVER_ROOT_URL=https://metrics.yourdomain.com
    volumes:
      - grafana_data:/var/lib/grafana
      - ./configs/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - aiml-network
```

### 4.2 Configure Alerting

Create `configs/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourdomain.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'team-emails'

receivers:
  - name: 'team-emails'
    email_configs:
      - to: 'team@yourdomain.com'
        headers:
          Subject: '🚨 Alert: {{ .GroupLabels.alertname }}'
```

---

## 💾 Step 5: Backup Strategy

### 5.1 Database Backup

```bash
# Create backup script
cat << 'EOF' > /opt/scripts/backup-db.sh
#!/bin/bash
BACKUP_DIR=/backups/database
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec aiml-postgres pg_dump -U aimluser aimlstudio | \
  gzip > $BACKUP_DIR/aimlstudio_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/aimlstudio_$DATE.sql.gz s3://your-bucket/backups/
EOF

chmod +x /opt/scripts/backup-db.sh

# Add to crontab
echo "0 2 * * * /opt/scripts/backup-db.sh" | sudo crontab -
```

### 5.2 Model Backup

```bash
# Sync models to S3
cat << 'EOF' > /opt/scripts/backup-models.sh
#!/bin/bash
aws s3 sync /data/models s3://your-bucket/models/ --delete
EOF

chmod +x /opt/scripts/backup-models.sh

# Add to crontab (daily at 3 AM)
echo "0 3 * * * /opt/scripts/backup-models.sh" | sudo crontab -
```

---

## 🔄 Step 6: CI/CD Pipeline

### 6.1 GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run Tests
        run: |
          python -m pytest tests/

      - name: Build Docker Images
        run: |
          docker build -t aimlstudio:${{ github.sha }} .

      - name: Push to Registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push aimlstudio:${{ github.sha }}

      - name: Deploy to Server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /opt/ai-ml-studio
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d
```

---

## 📈 Step 7: Performance Tuning

### 7.1 System Limits

```bash
# Edit limits
sudo nano /etc/security/limits.conf

# Add:
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
```

### 7.2 Kernel Parameters

```bash
# Edit sysctl
sudo nano /etc/sysctl.conf

# Add:
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535
vm.swappiness = 10

# Apply
sudo sysctl -p
```

---

## ✅ Step 8: Verification

### 8.1 Health Checks

```bash
# API Health
curl https://api.yourdomain.com/health

# GPU Check
docker exec aiml-backend-prod nvidia-smi

# Check logs
docker-compose -f docker-compose.prod.yml logs -f --tail=100

# Check metrics
curl http://localhost:9090/metrics
```

### 8.2 Load Testing

```bash
# Install Apache Bench
sudo apt install -y apache2-utils

# Test API
ab -n 1000 -c 10 https://api.yourdomain.com/health

# Or use k6
k6 run load-test.js
```

---

## 🆘 Troubleshooting

### Common Issues

**1. GPU not detected in container**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker daemon
sudo nano /etc/docker/daemon.json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
sudo systemctl restart docker
```

**2. Out of memory errors**
```bash
# Check memory usage
docker stats

# Increase swap (if needed)
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. Connection refused**
```bash
# Check if port is open
sudo netstat -tulpn | grep 8000

# Check firewall
sudo ufw status

# Check docker network
docker network inspect aiml-network
```

---

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Let's Encrypt](https://letsencrypt.org/getting-started/)

---

## ✅ Post-Deployment Checklist

- [ ] All services running
- [ ] Health checks passing
- [ ] SSL certificate valid
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Backups scheduled
- [ ] Firewall configured
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained

**Your production deployment is complete! 🚀**
