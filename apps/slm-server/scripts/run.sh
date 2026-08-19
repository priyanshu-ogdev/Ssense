#!/usr/bin/env bash
# ==============================================================================
# run.sh - Ssense Virtual SLM Server Orchestrator
# Safely manages .env generation, Docker state, and container lifecycles.
# ==============================================================================

# Strict mode for error handling
set -euo pipefail

# ANSI Colors for terminal UI
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}     Ssense SLM Server - Zero-Hop Orchestrator        ${NC}"
echo -e "${BLUE}======================================================${NC}\n"

# Navigate to the slm-server root directory (parent of the scripts folder)
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------------------
# STEP 1: SAFE ENVIRONMENT SEEDING
# ------------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[!] .env file missing. Generating secure cryptographic keys for first-time setup...${NC}"
    python3 -c "
import secrets
api_key = secrets.token_urlsafe(32)
hmac_secret = secrets.token_urlsafe(48)
env_content = f'''SSENSE_ENV=production
SSENSE_API_KEYS={api_key}
SSENSE_ENTERPRISE_API_KEYS={api_key}
SSENSE_HMAC_SECRET={hmac_secret}
SSENSE_ALLOWED_ORIGINS=*
SSENSE_MAX_QUEUE_DEPTH=5000
'''
with open('.env', 'w') as f: f.write(env_content)
print(f'✅ Generated .env!')
print(f'⚠️  IMPORTANT: Copy this API_KEY to use in your Edge Extension: {api_key}')
"
else:
    echo -e "${GREEN}[✓] Existing .env file found. Preserving cryptographic keys.${NC}"
fi

# ------------------------------------------------------------------------------
# STEP 2: DOCKER DAEMON CHECK
# ------------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[✗] FATAL: Docker is not installed or not in PATH.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}[✗] FATAL: Docker daemon is not running. Please start Docker Desktop/Daemon.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 3: CONTAINER LIFECYCLE MANAGEMENT
# ------------------------------------------------------------------------------
CONTAINER_NAME="ssense-slm-server"

echo -e "\n${BLUE}[i] Assessing container state...${NC}"

# Check if container exists in any state
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    
    # Check if it is actively running
    if docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
        echo -e "${GREEN}[✓] Container '${CONTAINER_NAME}' is already ONLINE and serving traffic.${NC}"
        echo -e "    View real-time logs with: ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
        exit 0
    else
        echo -e "${YELLOW}[i] Container exists but is STOPPED. Attempting standard startup...${NC}"
        if docker compose up -d; then
            echo -e "${GREEN}[✓] Server successfully brought back online.${NC}"
        else
            echo -e "${RED}[✗] Standard startup failed due to configuration or network conflicts.${NC}"
            read -p "Would you like to force a clean rebuild of the container? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}[i] Executing clean rebuild...${NC}"
                docker compose up --build -d
            else
                echo -e "${YELLOW}[!] Boot sequence aborted.${NC}"
                exit 1
            fi
        fi
    fi
else
    echo -e "${YELLOW}[i] Container not found. Initiating First-Time Build & Boot sequence...${NC}"
    if docker compose up --build -d; then
        echo -e "${GREEN}[✓] Server successfully built and deployed.${NC}"
    else
        echo -e "${RED}[✗] Build failed. Check the Docker error logs above.${NC}"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# STEP 4: DIAGNOSTICS & SUCCESS
# ------------------------------------------------------------------------------
echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}     Server Boot Sequence Completed Successfully!     ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "📍 ${BLUE}Health Check:${NC} http://localhost:80/health"
echo -e "📄 ${BLUE}Live Logs:${NC}    docker compose logs -f"
echo -e "🛑 ${BLUE}To Stop:${NC}      docker compose down\n"