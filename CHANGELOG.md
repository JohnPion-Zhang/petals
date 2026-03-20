# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-03-20

### Added

#### HTTP Client Module

- **`petals.client.http_client.HTTPClient`**: New unified LLM client using litellm
  - Unified interface for OpenAI, Anthropic, and custom OpenAI-compatible endpoints
  - Automatic protocol routing based on model name
  - Configurable timeouts and retry behavior
  - Token usage tracking

- **`petals.client.http_client.LLMResponse`**: Structured response dataclass
  - Contains content, model, usage statistics, and raw response
  - Standardized interface across all providers

#### Agent Orchestrator

- **`petals.client.agent.AgentOrchestrator`**: Full agent loop implementation
  - Integrates HTTP client, tool parser, executor, and context manager
  - Automatic tool call detection and execution
  - Maximum iteration limit to prevent infinite loops
  - Conversation history and tool history tracking
  - Runtime model switching support

#### Tool Calling System

- **`petals.client.tool_registry.ToolRegistry`**: Centralized tool registration
  - Register sync and async functions as tools
  - Optional JSON schema support for documentation
  - Automatic async function detection and awaiting

- **`petals.client.tool_executor.ToolExecutor`**: Tool execution engine
  - Parallel tool execution using asyncio
  - Configurable timeout per tool
  - Dependency-aware execution ordering
  - Generator pattern for streaming results

- **`petals.client.tool_parser.ToolParser`**: LLM output parsing
  - Parses `<tool_call>tool_name({...})</tool_call>` syntax
  - JSON and key=value argument formats
  - Strict and lenient parsing modes

- **`petals.client.context_manager.ContextManager`**: Token budget management
  - Smart context trimming preserving essential elements
  - Token estimation with tokenizer fallback
  - Automatic budget checking before API calls

- **`petals.client.data_structures`**: Supporting data structures
  - `Message`: Conversation message representation
  - `ContextWindow`: Full context with history management

#### Multi-Protocol Support

- OpenAI `/chat/completions` endpoint support
- OpenAI Responses API (`/v1/responses`) support
- Anthropic `/messages` endpoint support
- Automatic protocol detection and routing

### Dependencies

- Added `litellm` as a dependency for unified LLM API access

### Documentation

- New "HTTP Client and Agent Layer" section in README.md
- Code examples for agent orchestration, tool calling, and model providers
- API documentation for all new modules

[2.3.0]: https://github.com/bigscience-workshop/petals/releases/tag/v2.3.0
