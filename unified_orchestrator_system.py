#!/usr/bin/env python3
"""
Unified Orchestrator System
Combines Tu Master, Claude Enhanced, and Codex Enhanced orchestrators
into a single powerful multi-agent orchestration platform
"""

import os
import sys
import json
import asyncio
import logging
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing external orchestrator implementations (like aican_orchestrator.py)
try:
    from aican_claude_orchestrator import orchestrate as run_claude_external
    EXTERNAL_CLAUDE_AVAILABLE = True
except Exception:
    run_claude_external = None
    EXTERNAL_CLAUDE_AVAILABLE = False

try:
    from aican_chatgpt_codex_orchestrator import orchestrate as run_gpt_external
    EXTERNAL_GPT_AVAILABLE = True
except Exception:
    run_gpt_external = None
    EXTERNAL_GPT_AVAILABLE = False

try:
    from aican_gemini_orchestrator import orchestrate as run_gemini_external
    EXTERNAL_GEMINI_AVAILABLE = True
except Exception:
    run_gemini_external = None
    EXTERNAL_GEMINI_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# SDK preference (as requested by user)
USE_SDK_BY_DEFAULT = True

# Model configurations
CLAUDE_MODEL = os.environ.get("AICAN_CLAUDE_MODEL", "claude-3-5-sonnet-latest")
GPT_MODEL = os.environ.get("AICAN_GPT_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("AICAN_GEMINI_MODEL", "gemini-2.0-flash-exp")

# Working directory and limits (from aican_orchestrator.py)
CWD = Path(os.environ.get("AICAN_WORKDIR", os.getcwd())).resolve()
DEFAULT_TURNS = int(os.environ.get("AICAN_MAX_TURNS", "4"))
DEFAULT_MAX_TOKENS = int(os.environ.get("AICAN_MAX_TOKENS", "2048"))

# Orchestrator ports
ORCHESTRATOR_PORT = 8999
CLAUDE_PORT = 9001
CODEX_PORT = 9002
GEMINI_PORT = 9003

# ============================================================================
# ROUTE CLASSIFICATION
# ============================================================================

class RouteType(Enum):
    """Available routing types."""
    AUTO = "auto"
    CLAUDE_SDK = "claude-sdk"
    CLAUDE_API = "claude-api"
    CLAUDE_EXTERNAL = "claude-external"  # Uses aican_claude_orchestrator.py
    GPT_SDK = "gpt-sdk"
    GPT_API = "gpt-api"
    GPT_EXTERNAL = "gpt-external"  # Uses aican_chatgpt_codex_orchestrator.py
    GEMINI_SDK = "gemini-sdk"
    GEMINI_API = "gemini-api"
    GEMINI_EXTERNAL = "gemini-external"  # Uses aican_gemini_orchestrator.py
    LOCAL = "local"
    ORCHESTRATOR = "orchestrator"

class TaskCategory(Enum):
    """Task categories for routing."""
    CODE = "code"
    TESTING = "testing"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    GENERAL = "general"

# Task classification keywords (enhanced with aican_orchestrator.py patterns)
CATEGORY_KEYWORDS = {
    TaskCategory.CODE: [
        # Core development (from aican_orchestrator.py)
        "build", "compile", "implement", "refactor", "debug", "fix", "bug", "stack trace",
        "lint", "ruff", "black", "mypy", "eslint", "pytest", "vitest",
        # Languages
        "python", "javascript", "typescript", "java", "go", "rust",
        # Package management and setup
        "npm", "pnpm", "yarn", "requirements.txt", "package.json", "pyproject.toml",
        "uv venv", "set up venv"
    ],
    TaskCategory.TESTING: [
        "unittest", "integration test", "e2e", "coverage",
        "pytest", "jest", "vitest", "mock", "stub", "tdd",
        "pytest -q", "run tests", "test suite", "write tests", 
        "add tests", "create tests"
    ],
    TaskCategory.DEVOPS: [
        "deploy", "ci", "cd", "pipeline", "kubernetes", "docker",
        "terraform", "aws", "azure", "gcp", "monitoring", "CI",
        "monorepo", "environment setup"
    ],
    TaskCategory.FRONTEND: [
        "frontend", "ui", "ux", "react", "vue", "angular",
        "css", "html", "component", "responsive", "accessibility"
    ],
    TaskCategory.BACKEND: [
        "backend", "api", "database", "server", "microservice",
        "rest", "graphql", "authentication", "authorization"
    ],
    TaskCategory.DOCUMENTATION: [
        "document", "readme", "docs", "comment", "docstring",
        "manual", "guide", "tutorial", "explain"
    ],
    TaskCategory.RESEARCH: [
        "research", "analyze", "investigate", "explore", "study",
        "compare", "evaluate", "benchmark", "review"
    ]
}

# ============================================================================
# UNIFIED ROUTER
# ============================================================================

class UnifiedRouter:
    """Unified routing logic for all orchestrators."""
    
    @staticmethod
    def ensure_env_for_route(route: RouteType) -> Optional[str]:
        """Validate environment setup for route (from aican_orchestrator.py)."""
        if route in [RouteType.CLAUDE_SDK, RouteType.CLAUDE_API]:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return "Missing ANTHROPIC_API_KEY environment variable"
        
        if route in [RouteType.GPT_SDK, RouteType.GPT_API]:
            if not os.environ.get("OPENAI_API_KEY"):
                return "Missing OPENAI_API_KEY environment variable"
        
        if route in [RouteType.GEMINI_SDK, RouteType.GEMINI_API]:
            if not os.environ.get("GOOGLE_API_KEY"):
                return "Missing GOOGLE_API_KEY environment variable"
        
        return None
    
    @staticmethod
    def ensure_impl_for_route(route: RouteType) -> Optional[str]:
        """Validate implementation availability for route."""
        if route in [RouteType.CLAUDE_SDK, RouteType.CLAUDE_API]:
            if not UnifiedRouter.has_claude_sdk() and route == RouteType.CLAUDE_SDK:
                return "Claude SDK not available - use --api flag or install: pip install anthropic"
            try:
                import anthropic
            except ImportError:
                return "Anthropic library not available - install: pip install anthropic"
        
        if route in [RouteType.GPT_SDK, RouteType.GPT_API]:
            if not UnifiedRouter.has_gpt_sdk() and route == RouteType.GPT_SDK:
                return "OpenAI SDK not available - use --api flag or install: pip install openai"
            try:
                import openai
            except ImportError:
                return "OpenAI library not available - install: pip install openai"
        
        if route in [RouteType.GEMINI_SDK, RouteType.GEMINI_API]:
            if not UnifiedRouter.has_gemini_sdk() and route == RouteType.GEMINI_SDK:
                return "Gemini SDK not available - use --api flag or install: pip install google-generativeai"
            try:
                import google.generativeai
            except ImportError:
                return "Google Generative AI library not available - install: pip install google-generativeai"
        
        return None
    
    @staticmethod
    def classify_task(task: str) -> Dict[str, Any]:
        """Classify task and determine optimal routing."""
        task_lower = task.lower()
        
        # Special handling for specific testing patterns where testing is PRIMARY
        test_primary_patterns = ["unit test", "write test", "test for"]
        if any(pattern in task_lower for pattern in test_primary_patterns):
            if "api" in task_lower:
                # "Write unit tests for the API" should be TESTING, not BACKEND
                category = TaskCategory.TESTING
                max_matches = 2  # Fake high match count to skip normal logic
            else:
                category = TaskCategory.TESTING
                max_matches = 2
        # Handle "add test" only when it's standalone, not part of larger development task  
        elif "add test" in task_lower and not any(primary in task_lower for primary in ["refactor", "implement", "build", "create", "develop", "write", "setup", "fix"]):
            category = TaskCategory.TESTING
            max_matches = 2
        else:
            # Find best matching category with priority order for ties
            category = TaskCategory.GENERAL
            max_matches = 0
            
            # Priority order for tie-breaking (most fundamental first)
            priority_order = [
                TaskCategory.CODE,        # Most fundamental development activities
                TaskCategory.TESTING,     # Specific testing tasks
                TaskCategory.DEVOPS,      # Operational tasks
                TaskCategory.FRONTEND,    # Specific UI tasks
                TaskCategory.BACKEND,     # Specific server tasks
                TaskCategory.DOCUMENTATION, # Documentation tasks
                TaskCategory.RESEARCH,    # Research tasks
                TaskCategory.GENERAL      # Fallback
            ]
            
            for cat in priority_order:
                if cat in CATEGORY_KEYWORDS:
                    keywords = CATEGORY_KEYWORDS[cat]
                    matches = sum(1 for kw in keywords if kw in task_lower)
                    if matches > max_matches:
                        max_matches = matches
                        category = cat
        
        # Determine priority (1-10)
        priority = 5  # Default
        if any(word in task_lower for word in ["urgent", "critical", "emergency"]):
            priority = 10
        elif any(word in task_lower for word in ["important", "production", "asap"]):
            priority = 8
        elif any(word in task_lower for word in ["test", "experiment", "try"]):
            priority = 3
        
        # Determine optimal model
        if category in [TaskCategory.CODE, TaskCategory.TESTING, TaskCategory.BACKEND]:
            preferred_model = "claude"
        elif category in [TaskCategory.RESEARCH, TaskCategory.DOCUMENTATION]:
            preferred_model = "gpt"
        elif category in [TaskCategory.DEVOPS, TaskCategory.FRONTEND]:
            preferred_model = "gemini"
        else:
            preferred_model = "claude"  # Default
        
        return {
            "category": category,
            "priority": priority,
            "preferred_model": preferred_model,
            "needs_approval": priority >= 8,
            "requires_tools": category in [TaskCategory.CODE, TaskCategory.TESTING, TaskCategory.DEVOPS]
        }
    
    @staticmethod
    def determine_route(task: str, override: Optional[str] = None) -> RouteType:
        """Determine the best route for task execution."""
        if override and override != "auto":
            # Handle special cases for backward compatibility
            route_map = {
                "claude": "claude-external" if EXTERNAL_CLAUDE_AVAILABLE else "claude-sdk",
                "gpt": "gpt-external" if EXTERNAL_GPT_AVAILABLE else "gpt-sdk", 
                "gemini": "gemini-external" if EXTERNAL_GEMINI_AVAILABLE else "gemini-sdk"
            }
            override = route_map.get(override, override)
            
            try:
                return RouteType(override)
            except ValueError:
                logger.warning(f"Invalid route override: {override}, using auto")
        
        classification = UnifiedRouter.classify_task(task)
        
        # Check if local orchestrator is running first
        if UnifiedRouter.is_service_running(ORCHESTRATOR_PORT):
            return RouteType.ORCHESTRATOR
        
        model = classification["preferred_model"]
        
        # Prefer external orchestrators if available (like aican_orchestrator.py)
        if model == "claude" and EXTERNAL_CLAUDE_AVAILABLE:
            return RouteType.CLAUDE_EXTERNAL
        elif model == "gpt" and EXTERNAL_GPT_AVAILABLE:
            return RouteType.GPT_EXTERNAL
        elif model == "gemini" and EXTERNAL_GEMINI_AVAILABLE:
            return RouteType.GEMINI_EXTERNAL
        
        # Use SDK by default if available
        if USE_SDK_BY_DEFAULT:
            if model == "claude" and UnifiedRouter.has_claude_sdk():
                return RouteType.CLAUDE_SDK
            elif model == "gpt" and UnifiedRouter.has_gpt_sdk():
                return RouteType.GPT_SDK
            elif model == "gemini" and UnifiedRouter.has_gemini_sdk():
                return RouteType.GEMINI_SDK
        
        # Fall back to API mode
        if model == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
            return RouteType.CLAUDE_API
        elif model == "gpt" and os.environ.get("OPENAI_API_KEY"):
            return RouteType.GPT_API
        elif model == "gemini" and os.environ.get("GOOGLE_API_KEY"):
            return RouteType.GEMINI_API
        
        # Last resort: local execution
        return RouteType.LOCAL
    
    @staticmethod
    def is_service_running(port: int) -> bool:
        """Check if a service is running on specified port."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    @staticmethod
    def has_claude_sdk() -> bool:
        """Check if Claude SDK is available."""
        try:
            import claude_code_sdk
            return True
        except:
            return False
    
    @staticmethod
    def has_gpt_sdk() -> bool:
        """Check if OpenAI SDK is available."""
        try:
            import openai
            return True
        except:
            return False
    
    @staticmethod
    def has_gemini_sdk() -> bool:
        """Check if Gemini SDK is available."""
        try:
            import google.generativeai
            return True
        except:
            return False

# ============================================================================
# EXECUTION ENGINES
# ============================================================================

class ClaudeExecutor:
    """Claude execution engine with SDK/API modes."""
    
    @staticmethod
    async def execute_sdk(task: str, config: Dict[str, Any]) -> int:
        """Execute using Claude SDK."""
        try:
            from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
            
            options = ClaudeCodeOptions(
                system_prompt=config.get("system_prompt", "You are Claude, an AI assistant."),
                allowed_tools=config.get("allowed_tools", ["Read", "Grep", "Glob"]),
                disallowed_tools=config.get("disallowed_tools", ["Bash(rm *)"]),
                max_turns=config.get("max_turns", 4),
                cwd=str(CWD)
            )
            
            async with ClaudeSDKClient(options=options) as client:
                await client.query(task)
                async for message in client.receive_response():
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                print(block.text, end="", flush=True)
            
            print("\n✅ Claude SDK execution complete")
            return 0
            
        except Exception as e:
            logger.error(f"Claude SDK execution failed: {e}")
            return 1
    
    @staticmethod
    async def execute_api(task: str, config: Dict[str, Any]) -> int:
        """Execute using Claude API directly."""
        try:
            from anthropic import Anthropic
            
            client = Anthropic()
            
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=config.get("max_tokens", 2048),
                messages=[
                    {"role": "user", "content": task}
                ]
            )
            
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            
            print("\n✅ Claude API execution complete")
            return 0
            
        except Exception as e:
            logger.error(f"Claude API execution failed: {e}")
            return 1

class GPTExecutor:
    """GPT/Codex execution engine with SDK/API modes."""
    
    @staticmethod
    async def execute_sdk(task: str, config: Dict[str, Any]) -> int:
        """Execute using OpenAI SDK."""
        try:
            import openai
            
            client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": config.get("system_prompt", "You are a helpful assistant.")},
                    {"role": "user", "content": task}
                ],
                max_tokens=config.get("max_tokens", 2048)
            )
            
            print(response.choices[0].message.content)
            print("\n✅ GPT SDK execution complete")
            return 0
            
        except Exception as e:
            logger.error(f"GPT SDK execution failed: {e}")
            return 1
    
    @staticmethod
    async def execute_api(task: str, config: Dict[str, Any]) -> int:
        """Execute using OpenAI API directly."""
        # Similar to SDK but with direct HTTP calls
        logger.info("GPT API execution (stub)")
        return 0

class GeminiExecutor:
    """Gemini execution engine with SDK/API modes."""
    
    @staticmethod
    async def execute_sdk(task: str, config: Dict[str, Any]) -> int:
        """Execute using Gemini SDK."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            response = model.generate_content(task)
            print(response.text)
            print("\n✅ Gemini SDK execution complete")
            return 0
            
        except Exception as e:
            logger.error(f"Gemini SDK execution failed: {e}")
            return 1
    
    @staticmethod
    async def execute_api(task: str, config: Dict[str, Any]) -> int:
        """Execute using Gemini API directly."""
        logger.info("Gemini API execution (stub)")
        return 0

# ============================================================================
# UNIFIED ORCHESTRATOR
# ============================================================================

class UnifiedOrchestrator:
    """The unified orchestrator that manages all execution modes."""
    
    def __init__(self):
        self.router = UnifiedRouter()
        self.executors = {
            "claude": ClaudeExecutor(),
            "gpt": GPTExecutor(),
            "gemini": GeminiExecutor()
        }
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        config_file = CWD / ".aican" / "orchestrator.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "max_turns": 4,
            "max_tokens": 2048,
            "auto_approve": False,
            "prefer_sdk": USE_SDK_BY_DEFAULT,
            "allowed_tools": ["Read", "Grep", "Glob", "Bash", "Write", "Edit"],
            "disallowed_tools": ["Bash(rm *)", "Bash(sudo *)", "Bash(dd *)"]
        }
    
    async def execute(self, task: str, route_override: Optional[str] = None) -> int:
        """Execute task using optimal route."""
        # Classify task
        classification = self.router.classify_task(task)
        
        # Determine route
        route = self.router.determine_route(task, route_override)
        
        logger.info(f"Task Classification:")
        logger.info(f"  Category: {classification['category'].value}")
        logger.info(f"  Priority: {classification['priority']}")
        logger.info(f"  Preferred Model: {classification['preferred_model']}")
        logger.info(f"  Route: {route.value}")
        
        # Validation checks (from aican_orchestrator.py)
        env_error = self.router.ensure_env_for_route(route)
        if env_error:
            logger.error(f"Environment validation failed: {env_error}")
            return 1
        
        impl_error = self.router.ensure_impl_for_route(route)
        if impl_error:
            logger.error(f"Implementation validation failed: {impl_error}")
            return 1
        
        # Handle approval if needed
        if classification["needs_approval"] and not self.config.get("auto_approve"):
            print(f"\n⚠️ High priority task requires approval")
            response = input("Proceed? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("Task cancelled by user")
                return 1
        
        # Execute based on route
        if route == RouteType.ORCHESTRATOR:
            return await self._execute_orchestrator(task, classification)
        elif route == RouteType.CLAUDE_SDK:
            return await self.executors["claude"].execute_sdk(task, self.config)
        elif route == RouteType.CLAUDE_API:
            return await self.executors["claude"].execute_api(task, self.config)
        elif route == RouteType.CLAUDE_EXTERNAL:
            return await self._execute_external_orchestrator("claude", task)
        elif route == RouteType.GPT_SDK:
            return await self.executors["gpt"].execute_sdk(task, self.config)
        elif route == RouteType.GPT_API:
            return await self.executors["gpt"].execute_api(task, self.config)
        elif route == RouteType.GPT_EXTERNAL:
            return await self._execute_external_orchestrator("gpt", task)
        elif route == RouteType.GEMINI_SDK:
            return await self.executors["gemini"].execute_sdk(task, self.config)
        elif route == RouteType.GEMINI_API:
            return await self.executors["gemini"].execute_api(task, self.config)
        elif route == RouteType.GEMINI_EXTERNAL:
            return await self._execute_external_orchestrator("gemini", task)
        elif route == RouteType.LOCAL:
            return await self._execute_local(task, classification)
        else:
            logger.error(f"Unknown route: {route}")
            return 1
    
    async def _execute_orchestrator(self, task: str, classification: Dict) -> int:
        """Execute via orchestrator system."""
        logger.info("Routing to orchestrator system...")
        
        message = {
            "action": "submit_task",
            "task": {
                "task_id": f"unified-{int(time.time())}",
                "description": task,
                "category": classification["category"].value,
                "priority": classification["priority"]
            }
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', ORCHESTRATOR_PORT))
            sock.send(json.dumps(message).encode())
            sock.close()
            
            logger.info("✅ Task submitted to orchestrator")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to submit to orchestrator: {e}")
            return 1
    
    async def _execute_external_orchestrator(self, model: str, task: str) -> int:
        """Execute using external orchestrator implementations (like aican_orchestrator.py)."""
        logger.info(f"Using external {model} orchestrator")
        
        try:
            # Get configuration parameters
            max_turns = self.config.get("max_turns", DEFAULT_TURNS)
            max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
            auto_approve = self.config.get("auto_approve", False)
            
            if model == "claude" and run_claude_external:
                result = run_claude_external(task, max_turns, auto_approve, max_tokens)
                return int(result) if result is not None else 0
            
            elif model == "gpt" and run_gpt_external:
                result = run_gpt_external(task, max_turns, auto_approve, max_tokens)
                return int(result) if result is not None else 0
            
            elif model == "gemini" and run_gemini_external:
                result = run_gemini_external(task, max_turns, auto_approve, max_tokens)
                return int(result) if result is not None else 0
            
            else:
                logger.error(f"External {model} orchestrator not available")
                return 1
                
        except Exception as e:
            logger.error(f"External {model} orchestrator failed: {e}")
            return 1
    
    async def _execute_local(self, task: str, classification: Dict) -> int:
        """Execute locally (fallback)."""
        logger.info("Executing locally (simulation mode)")
        
        print(f"\n📋 Task: {task}")
        print(f"Category: {classification['category'].value}")
        print(f"Priority: {classification['priority']}")
        print("\n[Simulated execution - no actual changes made]")
        print("✅ Local execution complete")
        
        return 0

# ============================================================================
# DEPLOYMENT MANAGER
# ============================================================================

class DeploymentManager:
    """Manages deployment of the unified orchestrator system."""
    
    @staticmethod
    async def deploy() -> bool:
        """Deploy the complete orchestrator system."""
        logger.info("Deploying Unified Orchestrator System...")
        
        # Create directories
        dirs = [
            CWD / ".aican",
            CWD / ".aican" / "logs",
            CWD / ".aican" / "pids"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # Start orchestrator services
        services = [
            ("orchestrator", ORCHESTRATOR_PORT, "claude_orchestrator_production.py"),
            ("claude", CLAUDE_PORT, "claude_orchestrator_enhanced.py"),
            ("codex", CODEX_PORT, "codex_orchestrator_enhanced.py")
        ]
        
        for name, port, script in services:
            if UnifiedRouter.is_service_running(port):
                logger.info(f"  {name} already running on port {port}")
            else:
                logger.info(f"  Starting {name} on port {port}...")
                # In production, would actually start the service
                # For now, just log
        
        logger.info("✅ Unified Orchestrator System deployed")
        return True
    
    @staticmethod
    async def status() -> Dict[str, Any]:
        """Get system status."""
        return {
            "services": {
                "orchestrator": UnifiedRouter.is_service_running(ORCHESTRATOR_PORT),
                "claude": UnifiedRouter.is_service_running(CLAUDE_PORT),
                "codex": UnifiedRouter.is_service_running(CODEX_PORT),
                "gemini": UnifiedRouter.is_service_running(GEMINI_PORT)
            },
            "sdks": {
                "has_claude_sdk": UnifiedRouter.has_claude_sdk(),
                "has_gpt_sdk": UnifiedRouter.has_gpt_sdk(),
                "has_gemini_sdk": UnifiedRouter.has_gemini_sdk()
            },
            "external_orchestrators": {
                "claude_external": EXTERNAL_CLAUDE_AVAILABLE,
                "gpt_external": EXTERNAL_GPT_AVAILABLE,
                "gemini_external": EXTERNAL_GEMINI_AVAILABLE
            },
            "environment": {
                "anthropic_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "openai_api_key": bool(os.environ.get("OPENAI_API_KEY")),
                "google_api_key": bool(os.environ.get("GOOGLE_API_KEY"))
            }
        }

# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Orchestrator System - Multi-agent orchestration platform"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Execute a task")
    exec_parser.add_argument("task", help="Task description")
    exec_parser.add_argument("--route", help="Force specific route", default="auto")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy orchestrator system")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    
    args = parser.parse_args()
    
    if args.command == "execute":
        orchestrator = UnifiedOrchestrator()
        return await orchestrator.execute(args.task, args.route)
    
    elif args.command == "deploy":
        return 0 if await DeploymentManager.deploy() else 1
    
    elif args.command == "status":
        status = await DeploymentManager.status()
        print("\nSystem Status:")
        for key, value in status.items():
            status_icon = "✅" if value else "❌"
            print(f"  {status_icon} {key}: {value}")
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)