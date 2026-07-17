"""GBNF grammar builder for structured LLM output.

GBNF (GGML BNF) is llama.cpp's grammar format for constraining token
generation. On CPU this is nearly free — the grammar mask is applied
during sampling with no extra forward passes. This gives CPU inference
a structural advantage: guaranteed valid JSON tool calls without
retries or validation loops.
"""

from __future__ import annotations

from eigencore.agent.tool import Tool, ToolRegistry


class GBNFBuilder:
    """Builds GBNF grammars that constrain model output to valid structures."""

    # reusable primitives
    _WS = "ws ::= [ \\t\\n]*"
    _STRING = r'string ::= "\"" ([^"\\] | "\\" .)* "\""'
    _INTEGER = 'integer ::= "-"? [0-9]+'
    _NUMBER = 'number ::= "-"? [0-9]+ ("." [0-9]+)?'
    _BOOLEAN = 'boolean ::= "true" | "false"'

    @classmethod
    def tool_call_grammar(cls, registry: ToolRegistry) -> str:
        if not registry.tools:
            raise ValueError("Cannot build grammar for empty tool registry")

        rules: list[str] = []
        tool_alternatives: list[str] = []

        for tool in registry.tools:
            rule_name = f"call_{tool.name}"
            tool_alternatives.append(rule_name)
            rules.append(cls._tool_rule(tool, rule_name))

        root = 'root ::= ws "{" ws tool_call ws "}" ws'
        tool_union = "tool_call ::= " + " | ".join(tool_alternatives)

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

        param_pairs: list[str] = []
        for param in tool.parameters:
            type_rule = cls._type_to_rule(param.type)
            if param.enum:
                enum_alts = " | ".join(f'"\\"{v}\\""' for v in param.enum)
                type_rule = f"({enum_alts})"
            param_pair = f'"\\"{param.name}\\"" ws ":" ws {type_rule}'
            param_pairs.append(param_pair)

        args_body = ' ws "," ws '.join(param_pairs)
        args_part = f'"\\"args\\"" ws ":" ws "{{" ws {args_body} ws "}}"'
        return f'{rule_name} ::= {tool_name_part} ws "," ws {args_part}'

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
    def thought_or_action_grammar(cls, registry: ToolRegistry) -> str:
        """Grammar for ReAct: model must output either a thought or a tool call."""
        tool_grammar = cls.tool_call_grammar(registry)

        thought_rule = 'thought ::= "\\"thought\\"" ws ":" ws string'
        action_rule = 'action ::= "\\"action\\"" ws ":" ws "{" ws tool_call ws "}"'

        new_root = 'root ::= ws "{" ws (thought_block | action_block) ws "}" ws'
        thought_block = 'thought_block ::= "\\"type\\"" ws ":" ws "\\"thought\\"" ws "," ws thought'
        action_block = 'action_block ::= "\\"type\\"" ws ":" ws "\\"action\\"" ws "," ws action'

        remaining = "\n".join(tool_grammar.split("\n")[1:])
        parts = [new_root, thought_block, action_block, thought_rule, action_rule, remaining]
        return "\n".join(parts)
