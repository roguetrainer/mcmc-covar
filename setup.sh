#!/bin/bash
# ==============================================================================
# CoVaR MCMC Estimation - Virtual Environment Setup
# ==============================================================================
# This script creates a Python virtual environment and installs all dependencies
# needed to run both CoVaR implementations (Gibbs Sampler and PyMC/NUTS)
#
# Usage:
#   bash setup.sh
#   
# Or make executable and run:
#   chmod +x setup.sh
#   ./setup.sh
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="covar_mcmc_env"
PYTHON_MIN_VERSION="3.9"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    local python_cmd=$1
    local version=$($python_cmd --version 2>&1 | awk '{print $2}')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    
    if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# Main Setup
# ==============================================================================

print_header "CoVaR MCMC Estimation - Environment Setup"

# Step 1: Check for Python
print_info "Checking for Python installation..."

PYTHON_CMD=""
if check_command python3; then
    if check_python_version python3; then
        PYTHON_CMD="python3"
        print_success "Found Python 3.9+ at: $(which python3)"
        print_info "Version: $(python3 --version)"
    fi
elif check_command python; then
    if check_python_version python; then
        PYTHON_CMD="python"
        print_success "Found Python 3.9+ at: $(which python)"
        print_info "Version: $(python --version)"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3.9 or higher is required but not found."
    echo ""
    echo "Please install Python from:"
    echo "  • https://www.python.org/downloads/"
    echo "  • or via your system package manager"
    echo ""
    exit 1
fi

# Step 2: Check for pip
print_info "Checking for pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip is not installed or not working."
    echo ""
    echo "Please install pip:"
    echo "  • Ubuntu/Debian: sudo apt-get install python3-pip"
    echo "  • macOS: python3 -m ensurepip --upgrade"
    echo "  • Windows: python -m ensurepip --upgrade"
    echo ""
    exit 1
fi
print_success "pip is available"

# Step 3: Check for venv module
print_info "Checking for venv module..."
if ! $PYTHON_CMD -m venv --help &> /dev/null; then
    print_error "venv module is not available."
    echo ""
    echo "Please install venv:"
    echo "  • Ubuntu/Debian: sudo apt-get install python3-venv"
    echo "  • It should be included by default in Python 3.3+"
    echo ""
    exit 1
fi
print_success "venv module is available"

# Step 4: Create virtual environment
print_header "Creating Virtual Environment"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
        print_success "Removed existing environment"
    else
        print_info "Keeping existing environment and proceeding..."
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    print_info "Creating virtual environment: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    print_success "Virtual environment created"
else
    print_success "Using existing virtual environment"
fi

# Step 5: Activate virtual environment
print_info "Activating virtual environment..."

# Determine the activation script based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source "$VENV_NAME/Scripts/activate"
else
    # Linux/macOS
    source "$VENV_NAME/bin/activate"
fi

print_success "Virtual environment activated"

# Step 6: Upgrade pip
print_header "Upgrading pip, setuptools, and wheel"
python -m pip install --upgrade pip setuptools wheel
print_success "Core tools upgraded"

# Step 7: Install requirements
print_header "Installing Python Dependencies"

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory!"
    echo ""
    echo "Please ensure you're running this script from the directory"
    echo "containing requirements.txt"
    echo ""
    exit 1
fi

print_info "Installing packages from requirements.txt..."
echo ""

# Install with progress output
pip install -r requirements.txt

print_success "All dependencies installed successfully"

# Step 8: Verify installation
print_header "Verifying Installation"

echo ""
print_info "Checking critical packages..."

# Function to check package
check_package() {
    if python -c "import $1" 2>/dev/null; then
        local version=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "N/A")
        print_success "$1 (version: $version)"
        return 0
    else
        print_error "$1 - FAILED"
        return 1
    fi
}

FAILED=0

check_package "numpy" || FAILED=1
check_package "scipy" || FAILED=1
check_package "pandas" || FAILED=1
check_package "matplotlib" || FAILED=1
check_package "seaborn" || FAILED=1
check_package "tqdm" || FAILED=1
check_package "pymc" || FAILED=1
check_package "arviz" || FAILED=1

echo ""

if [ $FAILED -eq 0 ]; then
    print_success "All critical packages verified!"
else
    print_warning "Some packages failed verification. You may need to troubleshoot."
fi

# Step 9: Create activation helper scripts
print_header "Creating Helper Scripts"

# Create activation script for easy access
cat > activate_env.sh << 'ACTIVATE_EOF'
#!/bin/bash
# Quick activation script for CoVaR MCMC environment

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source covar_mcmc_env/Scripts/activate
else
    source covar_mcmc_env/bin/activate
fi

echo "CoVaR MCMC environment activated!"
echo ""
echo "Available scripts:"
echo "  • python covar_mcmc_estimation.py  (Gibbs Sampler)"
echo "  • python covar_pymc_estimation.py  (PyMC/NUTS)"
echo ""
echo "To deactivate: deactivate"
ACTIVATE_EOF

chmod +x activate_env.sh
print_success "Created activate_env.sh helper script"

# Create Windows batch file
cat > activate_env.bat << 'ACTIVATE_BAT'
@echo off
REM Quick activation script for CoVaR MCMC environment (Windows)

call covar_mcmc_env\Scripts\activate.bat

echo CoVaR MCMC environment activated!
echo.
echo Available scripts:
echo   - python covar_mcmc_estimation.py  (Gibbs Sampler)
echo   - python covar_pymc_estimation.py  (PyMC/NUTS)
echo.
echo To deactivate: deactivate
ACTIVATE_BAT

print_success "Created activate_env.bat helper script (Windows)"

# Step 10: Print completion message
echo ""
print_header "Setup Complete!"

cat << EOF

${GREEN}✓ Virtual environment ready!${NC}

${BLUE}Next Steps:${NC}

${YELLOW}1. Activate the environment:${NC}
   
   ${BLUE}Linux/macOS:${NC}
     source activate_env.sh
     
   ${BLUE}Windows (Command Prompt):${NC}
     activate_env.bat
     
   ${BLUE}Windows (PowerShell):${NC}
     .\\covar_mcmc_env\\Scripts\\Activate.ps1
   
   ${BLUE}Or manually:${NC}
     source covar_mcmc_env/bin/activate  # Linux/macOS
     covar_mcmc_env\\Scripts\\activate    # Windows

${YELLOW}2. Run the implementations:${NC}

   ${BLUE}Gibbs Sampler (Educational):${NC}
     python covar_mcmc_estimation.py
   
   ${BLUE}PyMC/NUTS (Production):${NC}
     python covar_pymc_estimation.py

${YELLOW}3. Explore with Jupyter (optional):${NC}
     jupyter notebook

${YELLOW}4. Deactivate when done:${NC}
     deactivate

${BLUE}Documentation:${NC}
  • README.md - Complete guide
  • COMPARISON_Gibbs_vs_PyMC.md - Detailed comparison

${GREEN}Happy MCMC sampling!${NC}

EOF

# Keep the environment activated for the user
echo "The virtual environment is now active in this shell."
echo "Run 'deactivate' to exit the environment."
echo ""
