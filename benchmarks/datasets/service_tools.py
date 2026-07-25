from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


class VirtualServiceWorld:
    """Disposable state for email, observation, memory, and generic native tools."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.initial = deepcopy(initial_state or {})
        self.state = deepcopy(initial_state or {})

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "email_list": self._email_list,
            "email_search": self._email_search,
            "email_read": self._email_read,
            "send_email": self._send_email,
            "memory": self._memory,
            "observation_read": self._observation_read,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            result = arguments.get("_simulated_result", {"status": "simulated", "tool": tool_name})
            return json.dumps(result, sort_keys=True)
        return json.dumps(handler(arguments), sort_keys=True)

    def state_diff(self) -> dict[str, Any]:
        changed = {
            key: deepcopy(value)
            for key, value in self.state.items()
            if self.initial.get(key) != value
        }
        removed = sorted(set(self.initial) - set(self.state))
        return {"changed": changed, "removed": removed}

    def _email_list(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"emails": deepcopy(self.state.get("emails", []))}

    def _email_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).casefold()
        matches = [
            email
            for email in self.state.get("emails", [])
            if query in json.dumps(email, sort_keys=True).casefold()
        ]
        return {"emails": deepcopy(matches)}

    def _email_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        identifier = str(arguments.get("email_id", ""))
        for email in self.state.get("emails", []):
            if str(email.get("id")) == identifier:
                return {"email": deepcopy(email)}
        return {"error": "email not found"}

    def _send_email(self, arguments: dict[str, Any]) -> dict[str, Any]:
        message = {
            "to": str(arguments.get("to", "")),
            "subject": str(arguments.get("subject", "")),
            "body": str(arguments.get("body", "")),
        }
        self.state.setdefault("outbox", []).append(message)
        return {"status": "queued_in_fixture_outbox", "message": message}

    def _observation_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        key = str(arguments.get("key", "default"))
        return {"key": key, "content": self.state.get("observations", {}).get(key, "")}

    def _memory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "view"))
        path = str(arguments.get("path", ""))
        memories = self.state.setdefault("memories", {})
        if command == "view":
            return {"path": path, "content": memories.get(path)}
        if command in {"create", "insert"}:
            memories[path] = str(arguments.get("content", ""))
        elif command == "str_replace":
            current = str(memories.get(path, ""))
            memories[path] = current.replace(
                str(arguments.get("old_str", "")),
                str(arguments.get("new_str", "")),
                1,
            )
        elif command == "delete":
            memories.pop(path, None)
        elif command == "rename":
            destination = str(arguments.get("new_path", ""))
            memories[destination] = memories.pop(path, "")
        else:
            return {"error": f"unsupported memory command: {command}"}
        return {"status": "updated", "state_diff": self.state_diff()}
