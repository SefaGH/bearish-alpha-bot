# Azure App Service Deployment Guide for Bearish Alpha Bot

This guide explains how to deploy the Bearish Alpha Bot to Azure App Service using the container we just fixed.

## 🔧 Prerequisites

1. **Azure CLI** installed and logged in
2. **Docker** installed locally
3. **Azure Container Registry** or Docker Hub access
4. **Azure subscription** with App Service access

## 📦 Container Deployment Steps

### 1. Build and Push Container

```bash
# Build the container with the fixed configuration
docker build -t bearishalphabot.azurecr.io/bearish-bot:v2 .

# Push to Azure Container Registry
docker push bearishalphabot.azurecr.io/bearish-bot:v2
```

### 2. Update Azure App Service

```bash
# Update the existing app service with the new container
az webapp config container set \
  --resource-group tradebot \
  --name BearishAlphaBot-Live-01 \
  --docker-custom-image-name bearishalphabot.azurecr.io/bearish-bot:v2
```

### 3. Configure Environment Variables

Copy these settings to your App Service Configuration > Application Settings:

```bash
# Core Trading Settings
TRADING_MODE=paper
DEBUG_MODE=false
ML_ENABLED=true
EXCHANGES=bingx

# Trading Parameters
CAPITAL_USDT=100
PER_TRADE_RISK_PCT=0.003
DAILY_MAX_TRADES=8
DUPLICATE_PREVENTION_THRESHOLD=0.0005

# Symbols
TRADING_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT
RSI_THRESHOLD_BTC=50
RSI_THRESHOLD_ETH=50
RSI_THRESHOLD_SOL=50

# Python Environment (Critical for Azure)
PYTHONUNBUFFERED=1
PYTHONPATH=/home/site/wwwroot:/home/site/wwwroot/src:/home/site/wwwroot/scripts
```

### 4. Add Secrets (if using live trading)

For live trading, add these as App Settings (they will be treated as secrets):

```bash
BINGX_KEY=your_api_key_here
BINGX_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🐛 Issues Fixed

### 1. ✅ ML Manifest TypeError Fixed
- **Problem**: `TypeError: expected str, bytes or os.PathLike object, not list`
- **Fix**: Added proper handling for `fallback_bundle` being a list in `production_coordinator.py`

### 2. ✅ Import Path Issues Resolved
- **Problem**: Module import errors due to incorrect PYTHONPATH
- **Fix**: Enhanced `azure_boot.py` with proper path setup and fallback imports

### 3. ✅ Azure Container Optimization
- **Problem**: Missing system dependencies and improper startup sequence
- **Fix**: Updated Dockerfile with TA-Lib installation and proper directory structure

### 4. ✅ Graceful Degradation
- **Problem**: Bot fails if ML components are missing
- **Fix**: Added fallback mechanisms and default GEMMA manifest creation

## 🔍 Monitoring and Troubleshooting

### Check Application Logs
```bash
az webapp log tail --resource-group tradebot --name BearishAlphaBot-Live-01
```

### Common Issues and Solutions

1. **Import Errors**
   - Ensure PYTHONPATH is set correctly in App Settings
   - Check that all required directories exist

2. **ML Model Missing**
   - Default manifest will be created automatically
   - Bot continues running with limited ML features

3. **WebSocket Connection Issues**
   - Verify exchange credentials (for live mode)
   - Check network connectivity from Azure

4. **Memory/CPU Issues**
   - Consider upgrading App Service plan
   - Enable debug mode temporarily to monitor resource usage

## 📊 Expected Startup Sequence

1. **Azure Boot** - Environment setup and path configuration
2. **Health Server** - (if keep_alive.py is available)
3. **Directory Creation** - Required folders and placeholder files
4. **ML Setup** - Default manifest creation and model verification
5. **Trading Bot Launch** - Actual bot execution with proper arguments

## 🎯 Production Checklist

- [ ] Container builds successfully
- [ ] All environment variables configured
- [ ] PYTHONPATH set correctly
- [ ] Exchange credentials added (for live trading)
- [ ] Resource group and subscription verified
- [ ] Monitoring and alerting configured
- [ ] Backup strategy for trading data

## 📈 Performance Optimization

For production use, consider:

1. **App Service Plan**: Use at least P1v2 for better performance
2. **Always On**: Enable to prevent cold starts
3. **Scaling**: Configure auto-scaling rules based on CPU/memory
4. **Monitoring**: Set up Application Insights for detailed telemetry

## 🔐 Security Best Practices

1. **Secrets Management**: Use Azure Key Vault for sensitive data
2. **Network Security**: Configure VNet integration if needed  
3. **Access Control**: Set up proper RBAC for the App Service
4. **Monitoring**: Enable security monitoring and alerts

---

The bot is now ready for Azure deployment with all critical issues resolved! 🚀