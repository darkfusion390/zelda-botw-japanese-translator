#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# start_mac.sh  —  Zelda BotW Translator  (macOS · Apple Vision OCR)
# ══════════════════════════════════════════════════════════════════════════════
# Run from the repo root:
#
#   chmod +x start_mac.sh     ← first time only
#   ./start_mac.sh
#
# Steps:
#   1. Check Python 3
#   2. Install missing Python dependencies
#   3. Check / install Ollama
#   4. Start Ollama serve if not already running
#   5. Pull qwen3:8b if not already downloaded
#   6. Run calibrate.py automatically if bounds.json is missing
#   7. Launch zelda_component_final_apple_ocr.py
# ══════════════════════════════════════════════════════════════════════════════

set -e

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Paths (all relative to repo root) ────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULARIZED="$REPO_ROOT/scripts/modularized"
COMPONENT="zelda_apple_ocr.py"
TRANSLATOR_SCRIPT="$MODULARIZED/$COMPONENT"
BOUNDS_FILE="$MODULARIZED/bounds.json"
CALIBRATE_SCRIPT="$MODULARIZED/calibrate.py"
OLLAMA_MODEL="qwen3:8b"

echo ""
echo -e "${BOLD}${CYAN}🎮  Zelda BotW Translator — macOS Startup (Apple Vision OCR)${NC}"
echo "════════════════════════════════════════════"

# ── Step 1: Python ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[1/6] Checking Python 3...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌  python3 not found.${NC}"
    echo -e "    Install from ${CYAN}https://python.org${NC} then re-run this script."
    exit 1
fi
echo -e "${GREEN}✅  $(python3 --version 2>&1)${NC}"

# ── Step 2: Python dependencies ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}[2/6] Checking Python dependencies...${NC}"

# Format: "pip_install_name:python_import_name"
# pyobjc packages are macOS-only (Apple Vision / Quartz bindings)
PACKAGES=(
    "opencv-python:cv2"
    "numpy:numpy"
    "requests:requests"
    "flask:flask"
    "pyobjc-framework-Vision:Vision"
    "pyobjc-framework-Quartz:Quartz"
    "fugashi:fugashi"
    "unidic-lite:unidic_lite"
    "pykakasi:pykakasi"
    "jamdict:jamdict"
)

MISSING=()
for entry in "${PACKAGES[@]}"; do
    pip_name="${entry%%:*}"
    import_name="${entry##*:}"
    if python3 -c "import $import_name" &>/dev/null 2>&1; then
        echo -e "  ${GREEN}✅  $pip_name${NC}"
    else
        echo -e "  ${YELLOW}⚠️   $pip_name — not found${NC}"
        MISSING+=("$pip_name")
    fi
done

# jamdict-data has no importable module — check via pip show
if python3 -m pip show jamdict-data &>/dev/null 2>&1; then
    echo -e "  ${GREEN}✅  jamdict-data${NC}"
else
    echo -e "  ${YELLOW}⚠️   jamdict-data — not found${NC}"
    MISSING+=("jamdict-data")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}📦  Installing ${#MISSING[@]} missing package(s)...${NC}"
    for pkg in "${MISSING[@]}"; do
        echo -e "    Installing $pkg..."
        pip3 install "$pkg" --quiet
        echo -e "    ${GREEN}✅  $pkg installed${NC}"
    done
    echo -e "${GREEN}✅  All dependencies installed${NC}"
else
    echo -e "${GREEN}✅  All dependencies present${NC}"
fi

# ── Step 3: Ollama ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/6] Checking Ollama...${NC}"
if ! command -v ollama &>/dev/null; then
    echo -e "${YELLOW}⚠️   Ollama not found.${NC}"
    if command -v brew &>/dev/null; then
        echo -e "    Installing via Homebrew..."
        brew install ollama
        echo -e "${GREEN}✅  Ollama installed${NC}"
    else
        echo -e "${RED}❌  Homebrew not found. Install Ollama manually:${NC}"
        echo -e "    ${CYAN}https://ollama.com${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅  Ollama $(ollama --version 2>&1 | head -1)${NC}"
fi

# ── Step 4: Ollama server ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[4/6] Starting Ollama server...${NC}"
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${GREEN}✅  Ollama already running${NC}"
else
    echo -e "  Starting ollama serve in background..."
    ollama serve > /tmp/ollama_zelda.log 2>&1 &
    OLLAMA_PID=$!
    # Poll up to 15 seconds
    STARTED=0
    for i in {1..15}; do
        sleep 1
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            STARTED=1
            echo -e "${GREEN}✅  Ollama server started (PID $OLLAMA_PID)${NC}"
            break
        fi
    done
    if [ $STARTED -eq 0 ]; then
        echo -e "${RED}❌  Ollama did not start within 15 seconds.${NC}"
        echo -e "    Check /tmp/ollama_zelda.log or run 'ollama serve' in a separate terminal."
        exit 1
    fi
fi

# ── Step 5: Pull model ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[5/6] Checking model $OLLAMA_MODEL...${NC}"
echo -e "  First run will download ~5GB — subsequent runs skip download"
if ! ollama pull "$OLLAMA_MODEL"; then
    echo -e "${RED}❌  Failed to pull $OLLAMA_MODEL. Check your internet connection.${NC}"
    exit 1
fi
echo -e "${GREEN}✅  $OLLAMA_MODEL ready${NC}"

# ── Step 6: bounds.json ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[6/6] Checking bounds.json...${NC}"
if [ -f "$BOUNDS_FILE" ]; then
    REGION_COUNT=$(python3 -c "import json; d=json.load(open('$BOUNDS_FILE')); print(len(d))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✅  bounds.json found — $REGION_COUNT region(s)${NC}"
else
    echo -e "${YELLOW}⚠️   bounds.json not found — launching calibrate.py...${NC}"
    echo ""
    echo -e "  ${CYAN}Instructions:${NC}"
    echo -e "  • A camera frame will open"
    echo -e "  • Draw a rectangle around each text region (dialogue box, item title, etc.)"
    echo -e "  • Press ENTER or SPACE to confirm each region and give it a name"
    echo -e "  • Press Q when you have defined all regions"
    echo ""
    # Disable set -e so calibrate exiting normally doesn't abort
    set +e
    cd "$MODULARIZED" && python3 calibrate.py
    set -e
    if [ ! -f "$BOUNDS_FILE" ]; then
        echo -e "${RED}❌  bounds.json was not created.${NC}"
        echo -e "    Make sure your camera is connected and re-run the script."
        exit 1
    fi
    REGION_COUNT=$(python3 -c "import json; d=json.load(open('$BOUNDS_FILE')); print(len(d))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✅  bounds.json created — $REGION_COUNT region(s) defined${NC}"
fi

# ── Verify component exists ───────────────────────────────────────────────────
if [ ! -f "$TRANSLATOR_SCRIPT" ]; then
    echo -e "${RED}❌  Component script not found: $TRANSLATOR_SCRIPT${NC}"
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo -e "════════════════════════════════════════════"
echo -e "${BOLD}${GREEN}🚀  Launching: $COMPONENT${NC}"
echo -e "    UI  →  ${CYAN}http://localhost:5002${NC}"
echo -e "    Stop  →  Ctrl+C"
echo -e "════════════════════════════════════════════"
echo ""

cd "$MODULARIZED"
python3 "$TRANSLATOR_SCRIPT"
