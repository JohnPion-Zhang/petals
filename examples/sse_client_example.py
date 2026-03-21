"""
Example: SSE Client for Tool Execution

Demonstrates how to consume the SSE streaming endpoint from a client.

Run the server first:
    python examples/sse_server_example.py

Then run this client:
    python examples/sse_client_example.py
"""
import asyncio
import httpx


async def main():
    """Main client example."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # --- Health Check ---
        print("=== Health Check ===")
        response = await client.get(f"{base_url}/api/v1/health")
        print(f"Health: {response.json()}")
        print()

        # --- Single Node Execution ---
        print("=== Single Node Execution ===")
        request_data = {
            "dag": {
                "nodes": [
                    {
                        "id": "add_numbers",
                        "name": "add",
                        "arguments": {"a": 10, "b": 20},
                    }
                ],
                "edges": [],
                "initial_args": {},
            }
        }

        async with client.stream("POST", f"{base_url}/api/v1/execute", json=request_data) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line:
                    print(f"  {line}")
        print()

        # --- Parallel Nodes Execution ---
        print("=== Parallel Nodes Execution ===")
        request_data = {
            "dag": {
                "nodes": [
                    {"id": "calc1", "name": "add", "arguments": {"a": 5, "b": 3}},
                    {"id": "calc2", "name": "multiply", "arguments": {"a": 4, "b": 7}},
                ],
                "edges": [],
                "initial_args": {},
            }
        }

        async with client.stream("POST", f"{base_url}/api/v1/execute", json=request_data) as response:
            async for line in response.aiter_lines():
                if line:
                    print(f"  {line}")
        print()

        # --- Dependency Chain Execution ---
        print("=== Dependency Chain Execution ===")
        request_data = {
            "dag": {
                "nodes": [
                    {"id": "step1", "name": "add", "arguments": {"a": 100, "b": 50}},
                    {
                        "id": "step2",
                        "name": "multiply",
                        "arguments": {"a": 0, "b": 0},
                        "dependencies": ["step1"],  # Depends on step1
                    },
                ],
                "edges": [],
                "initial_args": {},
            }
        }

        async with client.stream("POST", f"{base_url}/api/v1/execute", json=request_data) as response:
            async for line in response.aiter_lines():
                if line:
                    print(f"  {line}")
        print()

        # --- Batch Execution (Parallel) ---
        print("=== Batch Execution (Parallel) ===")
        request_data = {
            "dags": [
                {
                    "nodes": [{"id": "d1", "name": "add", "arguments": {"a": 1, "b": 2}}],
                    "edges": [],
                    "initial_args": {},
                },
                {
                    "nodes": [{"id": "d2", "name": "multiply", "arguments": {"a": 3, "b": 4}}],
                    "edges": [],
                    "initial_args": {},
                },
            ],
            "parallel": True,
        }

        response = await client.post(f"{base_url}/api/v1/execute/batch", json=request_data)
        print(f"Status: {response.status_code}")
        result = response.json()
        for i, dag_result in enumerate(result["results"]):
            print(f"  DAG {i+1}: success={dag_result['success']}, nodes={dag_result['nodes']}")
        print()

        # --- Complex DAG with Multiple Dependencies ---
        print("=== Complex DAG (Diamond Pattern) ===")
        request_data = {
            "dag": {
                "nodes": [
                    {"id": "root", "name": "add", "arguments": {"a": 10, "b": 5}},
                    {"id": "branch1", "name": "multiply", "arguments": {"a": 0, "b": 0}, "dependencies": ["root"]},
                    {"id": "branch2", "name": "multiply", "arguments": {"a": 0, "b": 0}, "dependencies": ["root"]},
                    {
                        "id": "merge",
                        "name": "process_data",
                        "arguments": {"data": "", "uppercase": True},
                        "dependencies": ["branch1", "branch2"],
                    },
                ],
                "edges": [],
                "initial_args": {},
            }
        }

        async with client.stream("POST", f"{base_url}/api/v1/execute", json=request_data) as response:
            async for line in response.aiter_lines():
                if line:
                    print(f"  {line}")


if __name__ == "__main__":
    print("SSE Client Example")
    print("=" * 50)
    asyncio.run(main())
