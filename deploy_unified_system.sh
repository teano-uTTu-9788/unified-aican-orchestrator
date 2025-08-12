#!/bin/bash
# Deploy Unified Orchestrator System

echo "🚀 Deploying Unified AiCan Orchestrator System"
echo "=============================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ~/.aican/{logs,pids,config}
mkdir -p ~/.claude_orchestrator/{logs,pids}

# Check for required files
echo "🔍 Checking required files..."
REQUIRED_FILES=(
    "unified_orchestrator_system.py"
    "claude_orchestrator_enhanced.py"
    "codex_orchestrator_enhanced.py"
    "tu_orchestrator_master.py"
    "aican"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file missing"
    fi
done

# Check for API keys
echo ""
echo "🔑 Checking API keys..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ✅ Claude API key found"
else
    echo "  ⚠️  Claude API key not found (set ANTHROPIC_API_KEY)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✅ OpenAI API key found"
else
    echo "  ⚠️  OpenAI API key not found (set OPENAI_API_KEY)"
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    echo "  ✅ Google API key found"
else
    echo "  ⚠️  Google API key not found (set GOOGLE_API_KEY)"
fi

# Deploy using the unified system
echo ""
echo "🚀 Starting deployment..."
python3 aican deploy

# Check status
echo ""
echo "📊 Checking system status..."
python3 aican status

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Usage examples:"
echo "  ./aican 'Write a hello world program in Python'"
echo "  ./aican 'Test my authentication module' --route claude"
echo "  ./aican 'Deploy to production' --route orchestrator"
echo "  ./aican status  # Check system status"
echo ""
echo "Single-name shortcuts:"
echo "  ./claude 'task'  # Claude orchestrator"
echo "  ./codex 'task'   # Codex orchestrator"  
echo "  ./gemma 'task'   # Gemma orchestrator"
echo "  ./tu 'task'      # Tu master orchestrator"
echo ""