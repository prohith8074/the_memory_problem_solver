#!/bin/bash

# Azure Deployment Script for RAG System
# This script deploys the RAG system to Azure without Docker/Kubernetes

set -e

# Configuration
RESOURCE_GROUP="RAG-System"
LOCATION="eastus"
APP_NAME="rag-system-$(date +%s)"
STORAGE_ACCOUNT="ragstorage$(date +%s | tail -c 5)"

echo "🚀 Deploying RAG System to Azure"
echo "================================="

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    exit 1
fi

# Login to Azure
echo "🔐 Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Create resource group
echo "📁 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create storage account for file uploads
echo "💾 Creating storage account..."
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --output table

# Create file share for uploads
STORAGE_KEY=$(az storage account keys list --account-name $STORAGE_ACCOUNT --query "[0].value" -o tsv)
az storage share create \
    --account-name $STORAGE_ACCOUNT \
    --name uploads \
    --output table

# Create Redis cache
echo "🗄️ Creating Redis cache..."
az redis create \
    --location $LOCATION \
    --name "rag-redis-$(date +%s)" \
    --resource-group $RESOURCE_GROUP \
    --sku Basic \
    --vm-size C1 \
    --output table

REDIS_HOST=$(az redis show --name "rag-redis-$(date +%s)" --resource-group $RESOURCE_GROUP --query "hostName" -o tsv)
REDIS_KEY=$(az redis list-keys --name "rag-redis-$(date +%s)" --resource-group $RESOURCE_GROUP --query "primaryKey" -o tsv)

# Create web app
echo "🌐 Creating web app..."
az webapp up \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --runtime "PYTHON:3.11" \
    --output table

# Set environment variables
echo "⚙️ Configuring environment variables..."
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --setting REDIS_HOST="$REDIS_HOST" \
             REDIS_PORT="6380" \
             REDIS_PASSWORD="$REDIS_KEY" \
             REDIS_DB="0" \
             QDRANT_URL="http://localhost:6333" \
             QDRANT_COLLECTION_NAME="rag_documents" \
             QDRANT_VECTOR_SIZE="1024" \
             CHUNK_SIZE="1000" \
             CHUNK_OVERLAP="200" \
             MAX_FILE_SIZE_MB="10" \
             LOG_LEVEL="INFO" \
             --output table

echo ""
echo "✅ Deployment completed successfully!"
echo "===================================="
echo "🌐 Web App URL: https://$APP_NAME.azurewebsites.net"
echo ""
echo "📋 Next steps:"
echo "1. Set your API keys in the Azure portal:"
echo "   - COHERE_API_KEY"
echo "   - GROQ_API_KEY"
echo "   - FIRECRAWL_API_KEY (optional)"
echo "   - OPIK_API_KEY (optional)"
echo ""
echo "2. Update QDRANT_URL if using external Qdrant instance"
echo ""
echo "3. Access your RAG system at the URL above"
echo ""
echo "🧹 To clean up resources:"
echo "   az group delete --name $RESOURCE_GROUP --yes"