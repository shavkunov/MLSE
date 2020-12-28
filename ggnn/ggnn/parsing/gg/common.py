from collections import defaultdict
from functools import reduce
from itertools import chain
from typing import *

from ..ast.common import AstNode, AstParser


class GatedGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: AstNode) -> ():
        if node not in self.nodes:
            self.nodes[node] = len(self.nodes)

    def add_edge(self, node_from: AstNode, node_to: AstNode, edge_type: str) -> ():
        self.edges.append((self.nodes[node_from], self.nodes[node_to], edge_type))

    def get_nodes_of_type(self, node_type: str) -> List[AstNode]:
        return [node for node in self.nodes if node.node_type == node_type]

    def get_edges_from(self, node: AstNode) -> List[Tuple[int, int, str]]:
        node_id = self.nodes.get(node, -1)
        return [edge for edge in self.edges if edge[0] == node_id]

    def get_edges_to(self, node: AstNode) -> List[Tuple[int, int, str]]:
        node_id = self.nodes.get(node, -1)
        return [edge for edge in self.edges if edge[1] == node_id]

    def get_edges_of_type(self, edge_type: str) -> List[Tuple[int, int, str]]:
        return [edge for edge in self.edges if edge[2] == edge_type]


class EdgeProcessor:

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {}

    def get_enter_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {}

    def get_exit_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {}

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {}

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return None

    def get_edges_names(self) -> List[str]:
        return []


class GatedGraphParser:
    def __init__(self, edge_processors: List[EdgeProcessor], ast_parser: AstParser):
        self.ast_parser = ast_parser
        self.edge_processors = edge_processors
        self.enter_node_callbacks = defaultdict(list)
        self.exit_node_callbacks = defaultdict(list)
        self.enter_child_callbacks = defaultdict(list)
        self.exit_child_callbacks = defaultdict(list)
        for i, edge_processor in enumerate(edge_processors):
            for node_type, callback in edge_processor.get_enter_node_callbacks().items():
                self.enter_node_callbacks[node_type].append((i, callback))
            for node_type, callback in edge_processor.get_exit_node_callbacks().items():
                self.exit_node_callbacks[node_type].append((i, callback))
            for node_type, callback in edge_processor.get_enter_child_callbacks().items():
                self.enter_child_callbacks[node_type].append((i, callback))
            for node_type, callback in edge_processor.get_exit_child_callbacks().items():
                self.exit_child_callbacks[node_type].append((i, callback))

    def parse_from_ast(self, root: AstNode) -> GatedGraph:
        cursor = root.walk()
        graph = GatedGraph()
        metas = [edge_processor.init_meta(root, graph) for edge_processor in self.edge_processors]

        def get_callbacks(callbacks_dict, node_type):
            return chain(callbacks_dict[node_type], callbacks_dict['ANY'])

        def enter_node():
            node = cursor.current_node()
            parent, child_index = cursor.current_parent(), cursor.current_child_index()
            graph.add_node(node)
            if parent is not None:
                for i, callback in get_callbacks(self.enter_child_callbacks, parent.node_type):
                    callback(parent, child_index, graph, metas[i])
            for i, callback in get_callbacks(self.enter_node_callbacks, node.node_type):
                callback(node, graph, metas[i])

        def exit_node():
            node = cursor.current_node()
            parent, child_index = cursor.current_parent(), cursor.current_child_index()
            for i, callback in get_callbacks(self.exit_node_callbacks, node.node_type):
                callback(node, graph, metas[i])
            if parent is not None:
                for i, callback in get_callbacks(self.exit_child_callbacks, parent.node_type):
                    callback(parent, child_index, graph, metas[i])

        def go_down():
            while cursor.goto_first_child():
                enter_node()

        def go_next():
            if cursor.goto_next_sibling():
                enter_node()
                go_down()
                return True
            return cursor.goto_parent()

        enter_node()
        go_down()

        while True:
            exit_node()
            if not go_next():
                return graph

    def parse(self, content):
        ast = self.ast_parser.parse(content)
        return ast, self.parse_from_ast(ast)

    def parse_file(self, p):
        with open(p, 'r', encoding='utf-8') as file:
            return self.parse(file.read())

    def get_edge_names(self):
        return sum([edge_processor.get_edges_names() for edge_processor in self.edge_processors], [])


class UsesScopedDictionary:
    def __init__(self):
        self.scopes = [UsesScopedDictionary.ScopeData()]

    def enter_scope(self):
        self.scopes.append(UsesScopedDictionary.ScopeData())

    def exit_scope(self):
        return self.scopes.pop()

    def report_declaration(self, name: str):
        self.scopes[-1].report_declaration(name)

    def report_uses(self, name: str, nodes: List[AstNode]) -> ():
        self.scopes[-1].report_uses(name, nodes)

    def report_probable_uses(self, name: str, nodes: List[AstNode]) -> ():
        self.scopes[-1].report_probable_uses(name, nodes)

    def get_uses(self, name):
        buffer = []
        for scope in reversed(self.scopes):
            if name in scope.local_var_uses:
                buffer.extend(scope.local_var_uses[name])
                break
            if name in scope.out_var_probable_uses:
                buffer.extend(scope.out_var_probable_uses[name])
            if name in scope.out_var_uses:
                buffer.extend(scope.out_var_uses[name])
                break
        return buffer

    def merge_scopes(self, scopes):
        for scope in scopes:
            for name, uses in scope.out_var_uses.items():
                self.scopes[-1].report_uses(name, uses)
            for name, uses in scope.out_var_probable_uses.items():
                self.scopes[-1].report_probable_uses(name, uses)

    def merge_optional_scopes(self, scopes):
        for scope in scopes:
            for name, uses in scope.out_var_uses.items():
                self.scopes[-1].report_probable_uses(name, uses)
            for name, uses in scope.out_var_probable_uses.items():
                self.scopes[-1].report_probable_uses(name, uses)

    def merge_alternative_scopes(self, scopes):
        if len(scopes) == 1:
            self.merge_scopes(scopes)
            return
        common_var = reduce(set.intersection, [scope.out_var_uses.keys() for scope in scopes],
                            set(scopes[0].out_var_uses.keys()))
        common_uses = defaultdict(list)
        common_probable_uses = defaultdict(list)
        for scope in scopes:
            for name, uses in scope.out_var_uses.items():
                if name in common_var:
                    common_uses[name].extend(uses)
                else:
                    self.scopes[-1].report_probable_uses(name, uses)
            for name, uses in scope.out_var_probable_uses.items():
                if name in common_var:
                    common_probable_uses[name].extend(uses)
                else:
                    self.scopes[-1].report_probable_uses(name, uses)
        for name in common_uses:
            self.scopes[-1].report_uses(name, common_uses[name])
            self.scopes[-1].report_probable_uses(name, common_probable_uses[name])

    class ScopeData:
        def __init__(self):
            self.local_var_uses = dict()
            self.out_var_uses = defaultdict(list)
            self.out_var_probable_uses = defaultdict(list)

        def report_declaration(self, name: str):
            self.local_var_uses[name] = []

        def report_uses(self, name: str, nodes: List[AstNode]) -> ():
            if name in self.local_var_uses:
                uses = self.local_var_uses[name]
                uses.clear()
                uses.extend(nodes)
            else:
                uses = self.out_var_uses[name]
                uses.clear()
                uses.extend(nodes)
                self.out_var_probable_uses[name].clear()

        def report_probable_uses(self, name: str, nodes: List[AstNode]) -> ():
            if name in self.local_var_uses:
                self.local_var_uses[name].extend(nodes)
            else:
                self.out_var_probable_uses[name].extend(nodes)
