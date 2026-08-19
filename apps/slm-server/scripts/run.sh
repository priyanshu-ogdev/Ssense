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
# STEP 1: SAFE ENVIRONMENT SEEDING & KEY ROTATION
# ------------------------------------------------------------------------------
echo -e "${BLUE}[i] Auditing Cryptographic Keys (.env)...${NC}"

# Python handles cross-platform file age checking (10 months ~ 300 days)
PYTHON_LOGIC=$(cat << 'EOF'
import os, time, secrets

ENV_FILE = '.env'
TEN_MONTHS_SEC = 300 * 24 * 60 * 60

def generate_keys():
    api_key = secrets.token_urlsafe(32)
    hmac_secret = secrets.token_urlsafe(48)
    env_content = f"""SSENSE_ENV=production
SSENSE_API_KEYS={api_key}
SSENSE_ENTERPRISE_API_KEYS={api_key}
SSENSE_HMAC_SECRET={hmac_secret}
SSENSE_ALLOWED_ORIGINS=*
SSENSE_MAX_QUEUE_DEPTH=5000
"""
    with open(ENV_FILE, 'w') as f:
        f.write(env_content)
    return api_key

if not os.path.exists(ENV_FILE):
    print("MISSING|" + generate_keys())
elif (time.time() - os.path.getmtime(ENV_FILE)) > TEN_MONTHS_SEC:
    print("EXPIRED|" + generate_keys())
else:
    print("VALID|NONE")
EOF
)

# Execute Python logic and parse the returned state
KEY_STATE=$(python3 -c "$PYTHON_LOGIC")
STATUS=$(echo "$KEY_STATE" | cut -d'|' -f1)
NEW_API_KEY=$(echo "$KEY_STATE" | cut -d'|' -f2)

if [ "$STATUS" == "MISSING" ]; then
    echo -e "${YELLOW}[!] .env file missing. Generating secure cryptographic keys for first-time setup...${NC}"
    echo -e "${GREEN}✅ Generated .env!${NC}"
    echo -e "${RED}⚠️  IMPORTANT: Copy this API_KEY to use in your Edge Extension: ${NEW_API_KEY}${NC}\n"
elif [ "$STATUS" == "EXPIRED" ]; then
    echo -e "${YELLOW}[!] SECURITY ALERT: Cryptographic keys are older than 10 months.${NC}"
    echo -e "${YELLOW}[!] Executing mandatory Automated Key Rotation...${NC}"
    echo -e "${GREEN}✅ Regenerated .env with fresh keys!${NC}"
    echo -e "${RED}⚠️  IMPORTANT: Your API Key changed! Update your Edge Extension: ${NEW_API_KEY}${NC}\n"
else
    echo -e "${GREEN}[✓] Existing .env file found and is within the 10-month validity period.${NC}\n"
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

echo -e "${BLUE}[i] Assessing container state...${NC}"

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    if docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
        # If the keys rotated while the container was running, Docker Compose needs to recreate it
        if [ "$STATUS" == "EXPIRED" ]; then
            echo -e "${YELLOW}[i] Keys were rotated. Forcing container recreation to ingest new secrets...${NC}"
            docker compose up -d
        else
            echo -e "${GREEN}[✓] Container '${CONTAINER_NAME}' is already ONLINE and serving traffic.${NC}"
            echo -e "    View real-time logs with: ${YELLOW}docker compose logs -f${NC}"
            exit 0
        fi
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