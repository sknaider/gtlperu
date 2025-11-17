#!/bin/bash
#
# AI ML Studio - Optimized Installation Script
# For: RTX 5090 + Ryzen 9 9950X + 128GB RAM + Linux
#

set -e

echo "========================================"
echo "AI ML Studio - Optimized Installation"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.10+ is installed
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $python_version detected"

# Check CUDA
echo -e "\n${YELLOW}Checking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓${NC} CUDA $cuda_version detected"
else
    echo -e "${YELLOW}⚠${NC} CUDA not detected. GPU acceleration may not work."
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment created"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓${NC} Pip upgraded"

# Install PyTorch with CUDA support (for RTX 5090)
echo -e "\n${YELLOW}Installing PyTorch with CUDA 12.1...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo -e "${GREEN}✓${NC} PyTorch installed"

# Install other dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Initialize directories
echo -e "\n${YELLOW}Initializing workspace...${NC}"
python cli.py init
echo -e "${GREEN}✓${NC} Workspace initialized"

# Configure environment
echo -e "\n${YELLOW}Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠${NC} Created .env file. Please edit it with your API keys!"
else
    echo -e "${GREEN}✓${NC} .env file already exists"
fi

# Test GPU
echo -e "\n${YELLOW}Testing GPU...${NC}"
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# Done
echo ""
echo "========================================"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Edit .env with your API keys"
echo "3. Run: python start.py"
echo ""
echo "Or use the CLI:"
echo "  python cli.py --help"
echo ""
