from __future__ import annotations
from typing import List
from tree_sitter import Language
from tree_sitter.binding import Parser


class AstNode:
    def __init__(self, node_id: int, node_type: str, token: [str, None], children: List, var_type: str = None):
        self.node_id = node_id
        self.node_type = node_type
        self.token = token
        self.var_type = var_type
        self.children = children

    def pretty_print(self, indent=1, **print_args):
        space = '\t' * indent
        text = f'{self.node_id}{space}{self.node_type}'
        if self.var_type is not None:
            text += f' ({self.token}: {self.var_type})'
        elif self.token is not None:
            text += f' ({self.token})'
        print(text, **print_args)
        for child in self.children:
            child.pretty_print(indent + 1, **print_args)

    def walk(self):
        return AstCursor(self)

    def is_leaf(self):
        return not self.children

    def deep_equals(self, node, ignore_id=False):
        if ignore_id and self.node_id != node.node_id:
            return False
        if self.node_type != node.node_type or \
                self.token != node.token or \
                self.var_type != node.var_type or \
                len(self.children) != len(node.children):
            return False

        for a, b in zip(self.children, node.children):
            if not a.deep_equals(b):
                return False
        return True

    def subtrees(self) -> List[AstNode]:
        buffer = []

        def scan_tree(node):
            buffer.append(node)
            for child in node.children:
                scan_tree(child)

        scan_tree(self)
        return buffer

    @staticmethod
    def to_json(node):
        return {
            'node_id': node.node_id,
            'node_type': node.node_type,
            'token': node.token,
            'var_type': node.var_type,
            'children': list(map(AstNode.to_json, node.children))
        }

    @staticmethod
    def from_json(json_dict):
        return AstNode(
            json_dict['node_id'],
            json_dict['node_type'],
            json_dict['token'],
            list(map(AstNode.from_json, json_dict['children'])),
            json_dict['var_type']
        )


class AstCursor:

    def __init__(self, root):
        self.parents_stack = [AstNode(-1, "FAKE_NODE", None, [root])]
        self.child_pointer_stack = [0]

    def current_node(self):
        return self.parents_stack[-1].children[self.child_pointer_stack[-1]]

    def current_parent(self) -> [AstNode, None]:
        return self.parents_stack[-1] if len(self.parents_stack) > 1 else None

    def current_child_index(self) -> [int, None]:
        return self.child_pointer_stack[-1] if len(self.parents_stack) > 1 else None

    def goto_first_child(self):
        node = self.current_node()
        if not node.children:
            return False
        self.parents_stack.append(node)
        self.child_pointer_stack.append(0)
        return True

    def goto_parent(self):
        if len(self.parents_stack) == 1:
            return False
        self.parents_stack.pop()
        self.child_pointer_stack.pop()
        return True

    def goto_next_sibling(self):
        if len(self.parents_stack[-1].children) > self.child_pointer_stack[-1] + 1:
            self.child_pointer_stack[-1] += 1
            return True
        return False


class AstParser:
    def __init__(self, language):
        self.ast_parser = Parser()
        self.ast_parser.set_language(Language('build/languages.so', language))

    def parse(self, content):
        ast = self.ast_parser.parse(bytes(content, encoding='utf-8'))
        meta = self.init_meta(ast, content)
        children_stack = [[]]
        cursor = ast.walk()

        def push_node():
            children_stack.append([])
            self.enter_node(cursor, meta)

        def pop_node():
            node = self.create_node(cursor, children_stack.pop(), meta)
            if node is not None:
                children_stack[-1].append(node)
            self.exit_node(cursor, meta)
            return node

        def go_down():
            while cursor.goto_first_child():
                push_node()

        def go_next():
            if cursor.goto_next_sibling():
                push_node()
                go_down()
                return True
            return cursor.goto_parent()

        push_node()
        go_down()

        while True:
            current_node = pop_node()
            if not go_next():
                return current_node

    def parse_file(self, p):
        with open(p, 'r', encoding='utf-8') as file:
            return self.parse(file.read())

    def init_meta(self, ast, content):
        return None

    def create_node(self, cursor, children, meta):
        raise NotImplementedError

    def enter_node(self, cursor, meta):
        pass

    def exit_node(self, cursor, meta):
        pass


class ScopedDictionary:
    def __init__(self):
        self.scopes = [dict()]

    def enter_scope(self):
        self.scopes.append(dict())

    def exit_scope(self):
        self.scopes.pop()

    def __getitem__(self, key):
        for scope in reversed(self.scopes):
            result = scope.get(key)
            if result is not None:
                return result
        return None

    def __setitem__(self, key, value):
        self.scopes[-1][key] = value

    def __contains__(self, key):
        for scope in reversed(self.scopes):
            if key in scope:
                return True
        return False
