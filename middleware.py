import os
import json
import re
import logging
import httpx
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("middleware")

app = FastAPI(title="Security Supervisor Middleware Proxy")

# CORS middleware for dashboards / client UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env file manually to keep it zero-dependency and clean
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# Security Guard Supervisor: Default to local Ollama
SUPERVISOR_URL = os.getenv("SUPERVISOR_URL", "http://localhost:11434")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "supervisor-proxy:latest")

# Upstream LLM providers: Default OpenAI to local Ollama (since it routes to cloud models)
UPSTREAM_OPENAI_URL = os.getenv("UPSTREAM_OPENAI_URL", "http://localhost:11434")
UPSTREAM_ANTHROPIC_URL = os.getenv("UPSTREAM_ANTHROPIC_URL", "https://api.anthropic.com")

MOCK_UPSTREAM = os.getenv("MOCK_UPSTREAM", "False").lower() in ("true", "1", "yes")

DECISION_RE = re.compile(r'\{"decision"\s*:\s*"(BLOCK|PASS)"\}', re.IGNORECASE)
SYSTEM_PROMPT = (
    "You are a security supervisor for an AI agent's tool execution pipeline. "
    "Analyse the proposed tool call against user preferences and agent state. "
    'Reply with only JSON: {"decision": "PASS"} or {"decision": "BLOCK"}.'
)

logger.info(f"Middleware initialized.")
logger.info(f" -> Supervisor Model: {SUPERVISOR_MODEL} @ {SUPERVISOR_URL}")
logger.info(f" -> Upstream OpenAI: {UPSTREAM_OPENAI_URL}")
logger.info(f" -> Upstream Anthropic: {UPSTREAM_ANTHROPIC_URL}")

def format_tool_call(name: str, arguments) -> str:
    """Format tool name and args into Python-like function signature representation."""
    if isinstance(arguments, dict):
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
    elif isinstance(arguments, list):
        args_str = ", ".join(repr(v) for v in arguments)
    else:
        args_str = repr(arguments)
    return f"{name}({args_str})"

async def check_action_with_supervisor(user_rule: str, agent_state: str, proposed_action: str) -> str:
    """Queries local/cloud security supervisor model to BLOCK or PASS a tool call."""
    # If the user_rule already contains the dataset prompt tags, pass it directly
    if "[USER_PREFERENCES]" in user_rule:
        prompt = user_rule
    else:
        prompt = (
            f"[USER_PREFERENCES]\n{user_rule}\n"
            f"[AGENT_STATE]\n{agent_state}\n"
            f"[PROPOSED_ACTION]\n{proposed_action}"
        )

    url = SUPERVISOR_URL.rstrip('/')
    headers = {}
    
    # Authenticate cloud calls only if targeting a remote supervisor
    if OLLAMA_API_KEY and "localhost" not in url and "127.0.0.1" not in url:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    # Resolve Ollama endpoint
    if "ollama.com" in url or "localhost" in url or "127.0.0.1" in url:
        if url.endswith("/v1"):
            endpoint = url.replace("/v1", "/api/chat")
        elif url.endswith("/api"):
            endpoint = f"{url}/chat"
        elif "/api/chat" not in url:
            endpoint = f"{url}/api/chat"
        else:
            endpoint = url
    else:
        # Generic OpenAI compatible endpoint
        if url.endswith("/v1"):
            endpoint = f"{url}/chat/completions"
        elif "/chat/completions" not in url:
            endpoint = f"{url}/v1/chat/completions"
        else:
            endpoint = url

    payload = {
        "model": SUPERVISOR_MODEL,
        "think": False,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"num_predict": 32, "temperature": 0.0},
    }

    if "chat/completions" in endpoint:
        payload = {
            "model": SUPERVISOR_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }

    logger.info(f"Querying supervisor endpoint: {endpoint}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(endpoint, json=payload, headers=headers, timeout=45.0)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] if "chat/completions" in endpoint else data["message"]["content"]
            logger.info(f"Supervisor response raw output: {content.strip()}")
            match = DECISION_RE.search(content)
            if match:
                return match.group(1).upper()
    except Exception as e:
        logger.error(f"Supervisor call failed: {e}")

    return "BLOCK"  # Fail-secure on error

def extract_anthropic_context(req_data: dict) -> tuple[str, str]:
    """Extract user instruction (preferences) and history (state) from Anthropic request."""
    system_val = req_data.get("system", "")
    system_text = "\n".join(b.get("text", "") for b in system_val if b.get("type") == "text") if isinstance(system_val, list) else str(system_val)

    messages = req_data.get("messages", [])
    first_user_content = ""
    history_lines = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_str = " ".join(
                b.get("text", "") if b.get("type") == "text" else
                f"[Proposed Tool Call: {format_tool_call(b.get('name',''), b.get('input',{}))}]" if b.get("type") == "tool_use" else
                f"[Tool Result: {b.get('content','')}]" if b.get("type") == "tool_result" else ""
                for b in content if isinstance(b, dict)
            )
        else:
            content_str = str(content or "")

        if role == "user" and not first_user_content:
            first_user_content = content_str

        history_lines.append(f"{role.capitalize()}: {content_str}")

    user_rule = f"System: {system_text}\nUser Task: {first_user_content}".strip()
    if not user_rule:
        user_rule = "No specific instructions provided."
    return user_rule, "\n".join(history_lines)

def extract_openai_context(req_data: dict) -> tuple[str, str]:
    """Extract user instruction (preferences) and history (state) from OpenAI request."""
    messages = req_data.get("messages", [])
    system_text = ""
    first_user_content = ""
    history_lines = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            content_str = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        else:
            content_str = str(content or "")

        if role == "system":
            system_text += f"\n{content_str}"
        elif role == "user" and not first_user_content:
            first_user_content = content_str

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    fn_args = {"raw_arguments": fn.get("arguments")}
                content_str += f"\n[Proposed Tool Call: {format_tool_call(fn_name, fn_args)}]"

        if role == "tool":
            content_str = f"[Tool Result: {content_str}]"

        history_lines.append(f"{role.capitalize()}: {content_str}")

    user_rule = f"System: {system_text.strip()}\nUser Task: {first_user_content}".strip()
    if not user_rule:
        user_rule = "No specific instructions provided."
    return user_rule, "\n".join(history_lines)

def get_mock_anthropic_response(model: str, messages: list) -> dict:
    """Simulate Anthropic response for testing without keys."""
    last_msg = messages[-1].get("content", "") if messages else ""
    last_msg_str = " ".join(b.get("text", "") for b in last_msg if b.get("type") == "text") if isinstance(last_msg, list) else str(last_msg)
    last_msg_lower = last_msg_str.lower()

    if "[PROPOSED_ACTION]" in last_msg_str:
        # Reconstruct dynamic tool call from supervisor dataset prompt
        match = re.search(r"\[PROPOSED_ACTION\]\n(.*)", last_msg_str, re.DOTALL)
        if match:
            action_str = match.group(1).strip()
            
            # Try parsing the (2) Action: / (3) Action Input: format (dataset format)
            action_name_match = re.search(r"\(2\) Action:\s*(\w+)", action_str)
            action_input_match = re.search(r"\(3\) Action Input:\s*(.*)", action_str)
            
            name = None
            arguments = {}
            if action_name_match and action_input_match:
                name = action_name_match.group(1).strip()
                args_json = action_input_match.group(1).strip()
                try:
                    arguments = json.loads(args_json)
                except Exception:
                    arguments = {"raw_input": args_json}
            else:
                # Try parsing standard python name(args) signature format
                action_match = re.match(r"(\w+)\((.*)\)", action_str)
                if action_match:
                    name = action_match.group(1)
                    args_raw = action_match.group(2)
                    for item in re.finditer(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", args_raw):
                        k = item.group(1)
                        v = item.group(2).strip("'\"")
                        arguments[k] = v
            
            if name:
                return {
                    "id": "msg_mock_id",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Invoking tool {name} from dataset..."},
                        {"type": "tool_use", "id": "tool_mock_dynamic", "name": name, "input": arguments}
                    ],
                    "model": model,
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }

    content = [{"type": "text", "text": "Hello, how can I assist you with your project files today?"}]
    return {
        "id": "msg_mock_id",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 24}
    }

def get_mock_openai_response(model: str, messages: list) -> dict:
    """Simulate OpenAI response for testing without keys."""
    last_msg = messages[-1].get("content", "") if messages else ""
    last_msg_str = " ".join(b.get("text", "") for b in last_msg if isinstance(b, dict) and b.get("type") == "text") if isinstance(last_msg, list) else str(last_msg)
    last_msg_lower = last_msg_str.lower()

    message = {"role": "assistant", "content": None}
    
    if "[PROPOSED_ACTION]" in last_msg_str:
        # Reconstruct dynamic tool call from supervisor dataset prompt
        match = re.search(r"\[PROPOSED_ACTION\]\n(.*)", last_msg_str, re.DOTALL)
        if match:
            action_str = match.group(1).strip()
            
            # Try parsing the (2) Action: / (3) Action Input: format (dataset format)
            action_name_match = re.search(r"\(2\) Action:\s*(\w+)", action_str)
            action_input_match = re.search(r"\(3\) Action Input:\s*(.*)", action_str)
            
            name = None
            arguments = {}
            if action_name_match and action_input_match:
                name = action_name_match.group(1).strip()
                args_json = action_input_match.group(1).strip()
                try:
                    arguments = json.loads(args_json)
                except Exception:
                    arguments = {"raw_input": args_json}
            else:
                # Try parsing standard python name(args) signature format
                action_match = re.match(r"(\w+)\((.*)\)", action_str)
                if action_match:
                    name = action_match.group(1)
                    args_raw = action_match.group(2)
                    for item in re.finditer(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", args_raw):
                        k = item.group(1)
                        v = item.group(2).strip("'\"")
                        arguments[k] = v
            
            if name:
                message["content"] = f"Invoking tool {name} from dataset."
                message["tool_calls"] = [{
                    "id": "call_mock_dynamic",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments)
                    }
                }]
                return {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1719523200,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "logprobs": None,
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45}
                }

    message["content"] = "How can I help you?"
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1719523200,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45}
    }

@app.post("/v1/messages")
async def proxy_anthropic(request: Request):
    req_json = await request.json()
    if req_json.get("stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not supported by the security middleware proxy."
        )

    # Context extraction
    user_rule, agent_state = extract_anthropic_context(req_json)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("x-api-key", "anthropic-version")}

    # Upstream communication (mocked or actual)
    if MOCK_UPSTREAM or headers.get("x-api-key", "").startswith("mock-"):
        logger.info("MOCK MODE: Simulating upstream Anthropic response")
        resp_data = get_mock_anthropic_response(req_json.get("model", "claude-mock"), req_json.get("messages", []))
    else:
        headers["content-type"] = "application/json"
        async with httpx.AsyncClient() as client:
            try:
                # Forward to configured Upstream Anthropic URL
                resp = await client.post(f"{UPSTREAM_ANTHROPIC_URL.rstrip('/')}/v1/messages", json=req_json, headers=headers, timeout=60.0)
                if resp.status_code != 200:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                resp_data = resp.json()
            except Exception as e:
                logger.error(f"Error forwarding to Anthropic: {e}")
                raise HTTPException(status_code=502, detail=f"Bad Gateway: Upstream connection error: {e}")

    # Security check on tool calls
    content_blocks = resp_data.get("content", [])
    for block in content_blocks:
        if block.get("type") == "tool_use":
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            proposed_action = format_tool_call(tool_name, tool_input)

            logger.info(f"Intercepted tool call: {proposed_action}")
            decision = await check_action_with_supervisor(user_rule, agent_state, proposed_action)
            logger.info(f"Supervisor decision: {decision}")

            if decision == "BLOCK":
                err_response = {
                    "type": "error",
                    "error": {
                        "type": "permission_error",
                        "message": f"Action blocked by security supervisor: {proposed_action}"
                    }
                }
                return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=err_response)

    return resp_data

@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    req_json = await request.json()
    if req_json.get("stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not supported by the security middleware proxy."
        )

    # Context extraction
    user_rule, agent_state = extract_openai_context(req_json)
    auth_header = request.headers.get("authorization", "")

    # Upstream communication (mocked or actual)
    if MOCK_UPSTREAM or auth_header.startswith("Bearer mock-"):
        logger.info("MOCK MODE: Simulating upstream OpenAI response")
        resp_data = get_mock_openai_response(req_json.get("model", "gpt-mock"), req_json.get("messages", []))
    else:
        headers = {"content-type": "application/json"}
        if auth_header:
            headers["authorization"] = auth_header
            
        async with httpx.AsyncClient() as client:
            try:
                # Forward to configured Upstream OpenAI URL (e.g. local Ollama gateway)
                resp = await client.post(f"{UPSTREAM_OPENAI_URL.rstrip('/')}/v1/chat/completions", json=req_json, headers=headers, timeout=60.0)
                if resp.status_code != 200:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                resp_data = resp.json()
            except Exception as e:
                logger.error(f"Error forwarding to OpenAI: {e}")
                raise HTTPException(status_code=502, detail=f"Bad Gateway: Upstream connection error: {e}")

    # Security check on tool calls
    choices = resp_data.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("type") == "function":
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    tool_input = {"raw_arguments": fn.get("arguments")}
                proposed_action = format_tool_call(tool_name, tool_input)

                logger.info(f"Intercepted tool call: {proposed_action}")
                decision = await check_action_with_supervisor(user_rule, agent_state, proposed_action)
                logger.info(f"Supervisor decision: {decision}")

                if decision == "BLOCK":
                    err_response = {
                        "error": {
                            "message": f"Action blocked by security supervisor: {proposed_action}",
                            "type": "permission_error",
                            "param": None,
                            "code": "action_blocked"
                        }
                    }
                    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=err_response)

    return resp_data

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("MIDDLEWARE_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
