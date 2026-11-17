"""GBNF grammar builder for structured LLM output.

GBNF (GGML BNF) is llama.cpp's grammar format for constraining token
generation. On CPU this is nearly free — the grammar mask is applied
during sampling with no extra forward passes. This gives CPU inference
a structural advantage: guaranteed valid JSON tool calls without
retries or validation loops.
"""

from __future__ import annotations

import re

from eigencore.agent.tool import Tool, ToolRegistry


class GBNFBuilder:
    """Builds GBNF grammars that constrain model output to valid structures."""

    # reusable primitives
    _WS = 'ws ::= | " " | "\\n" [ \\t]{0,8}'
    # JSON string: no raw control characters, standard escapes only (as in llama.cpp json.gbnf)
    _STRING = (
        r'string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""'
    )
    _INTEGER = 'integer ::= "-"? [0-9]{1,15}'
    _NUMBER = 'number ::= "-"? [0-9]{1,15} ("." [0-9]{1,15})?'
    _BOOLEAN = 'boolean ::= "true" | "false"'

    @classmethod
    def tool_call_grammar(cls, registry: ToolRegistry) -> str:
        if not registry.tools:
            raise ValueError("Cannot build grammar for empty tool registry")

        rules: list[str] = []
        tool_alternatives: list[str] = []

        for tool in registry.tools:
            rule_name = cls._rule_name(tool.name)
            tool_alternatives.append(rule_name)
            rules.append(cls._tool_rule(tool, rule_name))

        root = 'root ::= ws "{" ws tool-call ws "}" ws'
        tool_union = "tool-call ::= " + " | ".join(tool_alternatives)

        parts = [
            root,
            tool_union,
            *rules,
            cls._WS,
            cls._STRING,
            cls._INTEGER,
            cls._NUMBER,
            cls._BOOLEAN,
        ]
        return "\n".join(parts)

    @classmethod
    def json_object_grammar(cls, properties: dict[str, str]) -> str:
        """Build a grammar for a fixed JSON object shape.

        Args:
            properties: mapping of key name → type ("string", "integer", "number", "boolean")
        """
        if not properties:
            return 'root ::= ws "{" ws "}" ws\n' + cls._WS

        pairs: list[str] = []
        for key, val_type in properties.items():
            type_rule = cls._type_to_rule(val_type)
            pairs.append(f'"\\"" "{key}" "\\"" ws ":" ws {type_rule}')

        body = ' ws "," ws '.join(pairs)
        root = f'root ::= ws "{{" ws {body} ws "}}" ws'
        return "\n".join([root, cls._WS, cls._STRING, cls._INTEGER, cls._NUMBER, cls._BOOLEAN])

    @classmethod
    def _tool_rule(cls, tool: Tool, rule_name: str) -> str:
        name_literal = f'"\\"{tool.name}\\""'
        tool_name_part = f'"\\"tool\\"" ws ":" ws {name_literal}'

        if not tool.parameters:
            args_part = '"\\"args\\"" ws ":" ws "{" ws "}"'
            return f'{rule_name} ::= {tool_name_part} ws "," ws {args_part}'

        def pair(param) -> str:
            type_rule = cls._type_to_rule(param.type)
            if param.enum:
                type_rule = "(" + " | ".join(f'"\\"{v}\\""' for v in param.enum) + ")"
            return f'"\\"{param.name}\\"" ws ":" ws {type_rule}'

        required = [pair(p) for p in tool.parameters if p.required]
        optional = [pair(p) for p in tool.parameters if not p.required]
        sep = 'ws "," ws'
        if required:
            body = f" {sep} ".join(required) + "".join(f" ({sep} {o})?" for o in optional)
        else:
            # all optional: either "{}" or the optional keys in declared order
            chain = optional[0] + "".join(f" ({sep} {o})?" for o in optional[1:])
            body = f"({chain})?"
        args_part = f'"\\"args\\"" ws ":" ws "{{" ws {body} ws "}}"'
        return f'{rule_name} ::= {tool_name_part} ws "," ws {args_part}'

    @staticmethod
    def _rule_name(tool_name: str) -> str:
        """llama.cpp rule names allow only letters, digits and '-'."""
        return "call-" + re.sub(r"[^A-Za-z0-9-]", "-", tool_name)

    @staticmethod
    def _type_to_rule(type_name: str) -> str:
        mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
        }
        rule = mapping.get(type_name)
        if rule is None:
            raise ValueError(f"Unsupported type for GBNF: {type_name}")
        return rule

    @classmethod
    def react_grammar(cls, registry: ToolRegistry) -> str:
        """Grammar for one ReAct step, matching ReActAgent.parse_step exactly:

        {"type": "thought", "content": "..."}
        {"type": "action", "tool": "<name>", "args": {...}}
        {"type": "final", "content": "..."}
        """
        content = '"\\"content\\"" ws ":" ws string'
        lines = [
            'root ::= "{" ws (thought | action | final) ws "}"',
            f'thought ::= "\\"type\\"" ws ":" ws "\\"thought\\"" ws "," ws {content}',
            'action ::= "\\"type\\"" ws ":" ws "\\"action\\"" ws "," ws tool-call',
            f'final ::= "\\"type\\"" ws ":" ws "\\"final\\"" ws "," ws {content}',
        ]
        if registry.tools:
            tool_rules = []
            alternatives = []
            for tool in registry.tools:
                rule_name = cls._rule_name(tool.name)
                alternatives.append(rule_name)
                tool_rules.append(cls._tool_rule(tool, rule_name))
            lines.append("tool-call ::= " + " | ".join(alternatives))
            lines.extend(tool_rules)
        else:
            # no tools: the action branch can never match
            lines[0] = 'root ::= "{" ws (thought | final) ws "}"'
            lines.pop(2)
        lines += [cls._WS, cls._STRING, cls._INTEGER, cls._NUMBER, cls._BOOLEAN]
        return "\n".join(lines)

    @classmethod
    def thought_or_action_grammar(cls, registry: ToolRegistry) -> str:
        """Deprecated name for react_grammar (it now also allows "final")."""
        return cls.react_grammar(registry)
