"""
Model Context Protocol (MCP) Server for ALETHEX.
Enables Claude Desktop, Claude Code, Cursor, and any MCP client to natively
audit memory, resolve temporal contradictions, and filter stale facts.

Usage:
    python -m alethex.integrations.mcp_server
    Or in claude_desktop_config.json:
    {
      "mcpServers": {
        "alethex": {
          "command": "python",
          "args": ["-m", "alethex.integrations.mcp_server"]
        }
      }
    }
"""

import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Setup lightweight logging to stderr (stdout is reserved for JSON-RPC messages)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s [ALETHEX-MCP] %(levelname)s: %(message)s")
logger = logging.getLogger("alethex-mcp")

# Lazy-loaded engine
_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from alethex.api import ConsistencyEngine
        # Defaults to fast ONNX INT8 or CPU inference
        _ENGINE = ConsistencyEngine(device="cpu")
    return _ENGINE


def handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
          "tools": {}
        },
        "serverInfo": {
            "name": "alethex-temporal-memory",
            "version": "0.1.0"
        }
    }


def handle_tools_list() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "audit_memory_consistency",
                "description": (
                    "Audits a list of timestamped statements or memory notes about entities. "
                    "Detects temporal contradictions, identifies superseded historical facts, "
                    "and returns a Consistency Index (0.0 to 1.0) with an actionable conflict report."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "List of memory entries with text, optional timestamp (ISO format), and optional source_id.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "description": "The fact, statement, or memory entry."},
                                    "timestamp": {"type": "string", "description": "ISO timestamp when the statement was made."},
                                    "source_id": {"type": "string", "description": "Identifier or turn ID."}
                                },
                                "required": ["text"]
                            }
                        }
                    },
                    "required": ["documents"]
                }
            },
            {
                "name": "filter_reconciled_context",
                "description": (
                    "Filters retrieved memory chunks or RAG documents prior to answering a query. "
                    "Automatically suppresses obsolete/superseded facts and flags unresolved contradictions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunks": {
                            "type": "array",
                            "description": "List of memory chunks to reconcile.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique chunk ID"},
                                    "text": {"type": "string", "description": "Chunk content"},
                                    "timestamp": {"type": "string", "description": "Creation timestamp"}
                                },
                                "required": ["text"]
                            }
                        }
                    },
                    "required": ["chunks"]
                }
            },
            {
                "name": "check_fact_pair",
                "description": (
                    "Checks the logical and temporal relationship between two assertions about the same entity. "
                    "Returns whether they are consistent, superseded (one replaced the other), or conflicting."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "premise": {"type": "string", "description": "First assertion (e.g. 'Alice works at Google in London in Jan 2024')"},
                        "hypothesis": {"type": "string", "description": "Second assertion (e.g. 'Alice moved to SF to work at Anthropic in June 2024')"},
                        "premise_time": {"type": "string", "description": "Optional ISO timestamp for first assertion"},
                        "hypothesis_time": {"type": "string", "description": "Optional ISO timestamp for second assertion"}
                    },
                    "required": ["premise", "hypothesis"]
                }
            }
        ]
    }


def handle_tool_call(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    engine = get_engine()

    if name == "audit_memory_consistency":
        docs = arguments.get("documents", [])
        now_iso = datetime.now().isoformat()
        normalized_docs = []
        for i, d in enumerate(docs):
            normalized_docs.append({
                "source_id": d.get("source_id", f"doc_{i}"),
                "text": d.get("text", ""),
                "timestamp": d.get("timestamp", now_iso)
            })
        report = engine.check_documents(normalized_docs)
        return [{"type": "text", "text": json.dumps(report, indent=2)}]

    elif name == "filter_reconciled_context":
        chunks = arguments.get("chunks", [])
        now_iso = datetime.now().isoformat()
        normalized_chunks = []
        for i, c in enumerate(chunks):
            normalized_chunks.append({
                "id": c.get("id", f"chunk_{i}"),
                "text": c.get("text", ""),
                "timestamp": c.get("timestamp", now_iso)
            })
        results = engine.filter_context(normalized_chunks)
        valid_chunks = [r for r in results if r.get("is_valid", True)]
        superseded_chunks = [r for r in results if not r.get("is_valid", True)]
        output = {
            "valid_reconciled_context": valid_chunks,
            "suppressed_obsolete_chunks": superseded_chunks,
            "total_input": len(chunks),
            "surviving_count": len(valid_chunks)
        }
        return [{"type": "text", "text": json.dumps(output, indent=2)}]

    elif name == "check_fact_pair":
        premise = arguments.get("premise", "")
        hypothesis = arguments.get("hypothesis", "")
        t1_str = arguments.get("premise_time") or datetime(2024, 1, 1).isoformat()
        t2_str = arguments.get("hypothesis_time") or datetime(2024, 6, 1).isoformat()

        docs = [
            {"source_id": "fact_1", "text": premise, "timestamp": t1_str},
            {"source_id": "fact_2", "text": hypothesis, "timestamp": t2_str}
        ]
        res = engine.check_documents(docs)
        return [{"type": "text", "text": json.dumps(res, indent=2)}]

    else:
        raise ValueError(f"Unknown tool: {name}")


def main():
    logger.info("ALETHEX MCP Server started. Listening on stdio...")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            logger.error(f"Invalid JSON received: {e}")
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": handle_initialize(params)}
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": handle_tools_list()}
        elif method == "tools/call":
            try:
                tool_res = handle_tool_call(params.get("name"), params.get("arguments", {}))
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": tool_res}}
            except Exception as ex:
                logger.exception("Error executing tool")
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(ex)}}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
