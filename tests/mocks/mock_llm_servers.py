"""
Mock LLM servers for testing protocol switching.

Run individual servers:
    python -m tests.mocks.mock_llm_servers --port 18001 --server chat
    python -m tests.mocks.mock_llm_servers --port 18002 --server responses
    python -m tests.mocks.mock_llm_servers --port 18003 --server anthropic

Or run all together:
    python -m tests.mocks.mock_llm_servers --mode all

Requires Flask: pip install flask
"""
import argparse
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not installed. Install with: pip install flask")
    exit(1)


# ============================================================================
# Shared Mock Response Helpers
# ============================================================================

def make_openai_chat_response(
    content: str,
    model: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> Dict[str, Any]:
    """Create a mock OpenAI chat/completions response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_openai_responses_response(
    content: str,
    model: str,
) -> Dict[str, Any]:
    """Create a mock OpenAI responses API response."""
    return {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "status": "completed",
        "model": model,
        "created_at": datetime.now().isoformat() + "Z",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


def make_anthropic_response(
    content: str,
    model: str,
    thinking: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a mock Anthropic messages API response."""
    blocks = [{"type": "text", "text": content}]

    response: Dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
        },
    }

    # Add thinking block if provided (for reasoning models)
    if thinking:
        response["content"] = [
            {"type": "thinking", "thinking": thinking},
            {"type": "text", "text": content},
        ]

    return response


def extract_messages_content(messages: List[Dict]) -> str:
    """Extract text content from messages list."""
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
    return " ".join(parts)


# ============================================================================
# Server 1: OpenAI /v1/chat/completions
# ============================================================================

def create_openai_chat_app() -> Flask:
    """Create Flask app for OpenAI chat/completions mock."""
    app = Flask(__name__)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        body = request.get_json()
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        user_content = extract_messages_content(messages)
        response_text = f"[OpenAI Chat] Received: {user_content[:50]}..."

        return jsonify(make_openai_chat_response(
            content=response_text,
            model=model,
        ))

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "server": "openai-chat"})

    return app


# ============================================================================
# Server 2: OpenAI /v1/responses
# ============================================================================

def create_openai_responses_app() -> Flask:
    """Create Flask app for OpenAI responses API mock."""
    app = Flask(__name__)

    @app.route("/v1/responses", methods=["POST"])
    def responses():
        body = request.get_json()
        model = body.get("model", "unknown")
        input_data = body.get("input", [])

        user_content = extract_messages_content(input_data)
        response_text = f"[OpenAI Responses] Received: {user_content[:50]}..."

        return jsonify(make_openai_responses_response(
            content=response_text,
            model=model,
        ))

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "server": "openai-responses"})

    return app


# ============================================================================
# Server 3: Anthropic /v1/messages
# ============================================================================

def create_anthropic_app() -> Flask:
    """Create Flask app for Anthropic messages API mock."""
    app = Flask(__name__)

    @app.route("/v1/messages", methods=["POST"])
    def messages():
        x_api_key = request.headers.get("x-api-key")
        anthropic_version = request.headers.get("anthropic-version", "2023-06-01")

        if not x_api_key:
            return jsonify({"error": {"type": "authentication_error", "message": "Missing x-api-key header"}}), 401

        body = request.get_json()
        model = body.get("model", "unknown")
        messages_list = body.get("messages", [])
        system = body.get("system", "")
        max_tokens = body.get("max_tokens", 1024)

        user_content = extract_messages_content(messages_list)
        response_text = f"[Anthropic] Received: {user_content[:50]}..."

        # Simulate thinking for certain models
        thinking = None
        if "sonnet" in model.lower() or "haiku" in model.lower():
            thinking = "I should provide a helpful response based on the user's request."

        return jsonify(make_anthropic_response(
            content=response_text,
            model=model,
            thinking=thinking,
        ))

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "server": "anthropic-messages"})

    return app


# ============================================================================
# Server Runner Functions (module-level for multiprocessing compatibility)
# ============================================================================

def run_chat_server(host: str = "0.0.0.0", port: int = 18001) -> None:
    """Run the OpenAI Chat server. Module-level for multiprocessing."""
    app = create_openai_chat_app()
    app.run(host=host, port=port, debug=False, threaded=True)


def run_responses_server(host: str = "0.0.0.0", port: int = 18002) -> None:
    """Run the OpenAI Responses server. Module-level for multiprocessing."""
    app = create_openai_responses_app()
    app.run(host=host, port=port, debug=False, threaded=True)


def run_anthropic_server(host: str = "0.0.0.0", port: int = 18003) -> None:
    """Run the Anthropic Messages server. Module-level for multiprocessing."""
    app = create_anthropic_app()
    app.run(host=host, port=port, debug=False, threaded=True)


# ============================================================================
# CLI Entry Points
# ============================================================================

def main():
    import multiprocessing

    parser = argparse.ArgumentParser(description="Mock LLM Servers")
    parser.add_argument("--server", choices=["chat", "responses", "anthropic"],
                        default="chat", help="Which server to run")
    parser.add_argument("--port", type=int, default=18001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--mode", choices=["all"], help="Run all servers")
    args = parser.parse_args()

    if args.mode == "all":
        print("Starting all three mock servers:")
        print("  - OpenAI Chat: http://localhost:18001/v1/chat/completions")
        print("  - OpenAI Responses: http://localhost:18002/v1/responses")
        print("  - Anthropic: http://localhost:18003/v1/messages")
        print("\nPress Ctrl+C to stop all servers.")

        p1 = multiprocessing.Process(target=run_chat_server, args=(args.host, 18001))
        p2 = multiprocessing.Process(target=run_responses_server, args=(args.host, 18002))
        p3 = multiprocessing.Process(target=run_anthropic_server, args=(args.host, 18003))

        p1.start()
        p2.start()
        p3.start()

        try:
            p1.join()
            p2.join()
            p3.join()
        except KeyboardInterrupt:
            print("\nStopping servers...")
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p1.join()
            p2.join()
            p3.join()
            print("Servers stopped.")
        return

    if args.server == "chat":
        app = create_openai_chat_app()
        print(f"Starting OpenAI Chat/Completions mock on {args.host}:{args.port}")
        print(f"Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    elif args.server == "responses":
        app = create_openai_responses_app()
        print(f"Starting OpenAI Responses mock on {args.host}:{args.port}")
        print(f"Endpoint: http://{args.host}:{args.port}/v1/responses")
    else:  # anthropic
        app = create_anthropic_app()
        print(f"Starting Anthropic Messages mock on {args.host}:{args.port}")
        print(f"Endpoint: http://{args.host}:{args.port}/v1/messages")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
