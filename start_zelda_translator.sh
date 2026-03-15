#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# start_zelda_translator.sh
# ══════════════════════════════════════════════════════════════════════════════
# Startup script for Zelda Translator (NLP version).
# Run this from the terminal after booting your MacBook:
#
#   chmod +x start_zelda_translator.sh   (first time only)
#   ./start_zelda_translator.sh
#
# What this script does:
#   1. Checks all Python dependencies and installs any that are missing
#   2. Checks that jamdict-data is installed (separate from jamdict library)
#   3. Verifies Ollama is installed
#   4. Starts Ollama serve in the background if it isn't already running
#   5. Pulls qwen2.5:7b if not already downloaded (skips if up to date)
#   6. Checks bounds.json exists — warns you if not
#   7. Reminds you to check your phone IP address
#   8. Launches the translator script
#
# Prerequisites:
#   - Ollama installed (https://ollama.com)
#   - IP Webcam app running on your S21 on the same WiFi network
#   - bounds.json created by calibrate.py (run that first if missing)
# ══════════════════════════════════════════════════════════════════════════════

set -e  # exit on any unexpected error

# ── Colours for terminal output ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

# ── Config — update these to match your setup ──────────────────────────────────
# Path to your translator script — defaults to same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"
TRANSLATOR_SCRIPT="$SCRIPT_DIR/zelda_translator_working_nlp.py"
OLLAMA_MODEL="qwen2.5:7b"
BOUNDS_FILE="$SCRIPT_DIR/bounds.json"

echo ""
echo -e "${BOLD}${CYAN}🎮  Zelda Translator — Startup${NC}"
echo "════════════════════════════════════════"

# ── Step 1: Check Python is available ─────────────────────────────────────────
echo ""
echo -e "${BOLD}[1/6] Checking Python...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌  python3 not found. Install it from https://python.org${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}✅  $PYTHON_VERSION${NC}"

# ── Step 2: Check and install Python dependencies ─────────────────────────────
echo ""
echo -e "${BOLD}[2/6] Checking Python dependencies...${NC}"

# Full list of required packages
# pyobjc-framework-Vision and pyobjc-framework-Quartz are macOS-only Apple OCR bindings
PACKAGES=(
    "opencv-python"
    "numpy"
    "requests"
    "flask"
    "pyobjc-framework-Vision"
    "pyobjc-framework-Quartz"
    "fugashi"
    "unidic-lite"
    "pykakasi"
    "jamdict"
    "jamdict-data"
)

# Map from pip package name to the Python import name used to check if it's installed.
# Bash 3.2 (default on macOS) doesn't support associative arrays so we use a function.
get_import_name() {
    case "$1" in
        "opencv-python")             echo "cv2" ;;
        "pyobjc-framework-Vision")   echo "Vision" ;;
        "pyobjc-framework-Quartz")   echo "Quartz" ;;
        # These match their import name directly
        *)                           echo "$1" ;;
    esac
}

MISSING=()

for pkg in "${PACKAGES[@]}"; do

    # jamdict-data has no importable module — check via pip show instead
    if [[ "$pkg" == "jamdict-data" ]]; then
        if python3 -m pip show jamdict-data &>/dev/null 2>&1; then
            echo -e "  ${GREEN}✅  jamdict-data${NC}"
        else
            echo -e "  ${YELLOW}⚠️   jamdict-data — missing${NC}"
            MISSING+=("$pkg")
        fi
        continue
    fi

    import_name=$(get_import_name "$pkg")

    if python3 -c "import $import_name" &>/dev/null 2>&1; then
        echo -e "  ${GREEN}✅  $pkg${NC}"
    else
        echo -e "  ${YELLOW}⚠️   $pkg — missing${NC}"
        MISSING+=("$pkg")
    fi
done

# Install anything missing
if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing packages: ${MISSING[*]}${NC}"
    for pkg in "${MISSING[@]}"; do
        echo -e "  📦  Installing $pkg..."
        pip3 install "$pkg" --quiet
        echo -e "  ${GREEN}✅  $pkg installed${NC}"
    done
else
    echo -e "${GREEN}✅  All Python dependencies present${NC}"
fi

# ── Step 3: Check Ollama is installed ─────────────────────────────────────────
echo ""
echo -e "${BOLD}[3/6] Checking Ollama...${NC}"
if ! command -v ollama &>/dev/null; then
    echo -e "${RED}❌  Ollama not found.${NC}"
    echo -e "    Install it from ${CYAN}https://ollama.com${NC} then re-run this script."
    exit 1
fi
echo -e "${GREEN}✅  Ollama installed${NC}"

# ── Step 4: Start Ollama serve if not already running ─────────────────────────
echo ""
echo -e "${BOLD}[4/6] Starting Ollama server...${NC}"
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${GREEN}✅  Ollama already running${NC}"
else
    echo -e "  🚀  Starting ollama serve in background..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Give it a moment to come up
    sleep 3
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo -e "${GREEN}✅  Ollama server started (PID $OLLAMA_PID)${NC}"
    else
        echo -e "${RED}❌  Ollama server failed to start. Try running 'ollama serve' manually.${NC}"
        exit 1
    fi
fi

# ── Step 5: Pull model if needed ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}[5/6] Checking model $OLLAMA_MODEL...${NC}"
echo -e "  (skips download if already up to date)"
ollama pull "$OLLAMA_MODEL"
echo -e "${GREEN}✅  $OLLAMA_MODEL ready${NC}"

# ── Step 6: Check bounds.json ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}[6/6] Checking bounds.json...${NC}"
if [ -f "$BOUNDS_FILE" ]; then
    echo -e "${GREEN}✅  bounds.json found${NC}"
    # Show the bounds so the user can confirm they look right
    echo -e "    $(cat $BOUNDS_FILE)"
else
    echo -e "${YELLOW}⚠️   bounds.json not found at $BOUNDS_FILE${NC}"
    echo -e "    Run ${CYAN}python3 calibrate.py${NC} first to define your dialogue box crop region."
    echo -e "    The translator will exit immediately without it."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Run calibrate.py first."
        exit 1
    fi
fi

# ── Phone reminder ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}📱  Reminder: Make sure IP Webcam is running on your S21${NC}"
echo -e "    Check ${CYAN}IP_WEBCAM_URL${NC} in the script matches your phone's current IP."
echo -e "    (Phone IPs can change after restarting if not set to static)"
echo ""

# ── Check translator script exists ────────────────────────────────────────────
if [ ! -f "$TRANSLATOR_SCRIPT" ]; then
    echo -e "${RED}❌  Translator script not found at:${NC}"
    echo -e "    $TRANSLATOR_SCRIPT"
    echo -e "    Update the TRANSLATOR_SCRIPT variable at the top of this file."
    exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}🚀  Launching Zelda Translator...${NC}"
echo -e "    UI will be available at ${CYAN}http://localhost:5002${NC}"
echo ""
echo "════════════════════════════════════════"
echo ""

python3 "$TRANSLATOR_SCRIPT"
