from typing import *

from ..ast.common import AstNode, ScopedDictionary
from ..ast.java_parser import JavaParser
from ..gg.common import *


class JavaGatedGraphParser(GatedGraphParser):
    def __init__(self):
        super().__init__(
            [
                SinkEdgeProcessor(),
                ChildEdgeProcessor(),
                NextLeafEdgeProcessor(),
                ReturnEdgeProcessor(),
                CalculatedFromEdgeProcessor(),
                LastLexicalUseEdgeProcessor(),
                LastReadWriteEdgeProcessor(),
                GuardedByEdgeProcessor()
            ],
            JavaParser()
        )


class SinkEdgeProcessor(EdgeProcessor):
    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {"ANY": SinkEdgeProcessor.__add_sink_edge}

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        sink_node = AstNode(-1, "SINK_NODE", None, [root])
        graph.add_node(sink_node)
        return sink_node

    def get_edges_names(self) -> List[str]:
        return ["TO_SINK_EDGE", "FROM_SINK_EDGE"]

    @staticmethod
    def __add_sink_edge(node: AstNode, graph: GatedGraph, meta: Any):
        graph.add_edge(node, meta, "TO_SINK_EDGE")
        graph.add_edge(meta, node, "FROM_SINK_EDGE")


class ChildEdgeProcessor(EdgeProcessor):
    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {}

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {"ANY": ChildEdgeProcessor.__add_child_edge}

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return None

    def get_edges_names(self) -> List[str]:
        return ["CHILD_EDGE", "PARENT_EDGE"]

    @staticmethod
    def __add_child_edge(node: AstNode, graph: GatedGraph, meta: Any):
        for child in node.children:
            graph.add_edge(node, child, "CHILD_EDGE")
            graph.add_edge(child, node, "PARENT_EDGE")


class NextLeafEdgeProcessor(EdgeProcessor):

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {}

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {"ANY": NextLeafEdgeProcessor.__add_next_leaf_edge}

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return NodeHolder()

    def get_edges_names(self) -> List[str]:
        return ["NEXT_LEAF_EDGE", "PREV_LEAF_EDGE"]

    @staticmethod
    def __add_next_leaf_edge(node: AstNode, graph: GatedGraph, meta: Any):
        if node.children:
            return
        if meta.node:
            graph.add_edge(node, meta.node, "PREV_LEAF_EDGE")
            graph.add_edge(meta.node, node, "NEXT_LEAF_EDGE")
        meta.node = node


class ReturnEdgeProcessor(EdgeProcessor):

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {"method_declaration": ReturnEdgeProcessor.__push_method_declaration}

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "method_declaration": ReturnEdgeProcessor.__pop_method_declaration,
            "return_statement": ReturnEdgeProcessor.__add_return_edge
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return []

    def get_edges_names(self) -> List[str]:
        return ["RETURN_FROM_EDGE", "TO_RETURN_EDGE"]

    @staticmethod
    def __push_method_declaration(node: AstNode, graph: GatedGraph, meta: Any):
        meta.append(node)

    @staticmethod
    def __pop_method_declaration(node: AstNode, graph: GatedGraph, meta: Any):
        meta.pop()

    @staticmethod
    def __add_return_edge(node: AstNode, graph: GatedGraph, meta: Any):
        if meta:
            graph.add_edge(node, meta[-1], "RETURN_FROM_EDGE")
            graph.add_edge(meta[-1], node, "TO_RETURN_EDGE")


class CalculatedFromEdgeProcessor(EdgeProcessor):

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "plus_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "minus_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "multiply_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "divide_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "remainder_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "xor_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "and_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "or_assignment_expression": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "variable_declarator": CalculatedFromEdgeProcessor.__pop_variable_assignment,
            "variable_identifier": CalculatedFromEdgeProcessor.__add_calculated_from_edge
        }

    def get_exit_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "plus_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "minus_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "multiply_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "divide_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "remainder_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "xor_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "and_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "or_assignment_expression": CalculatedFromEdgeProcessor.__push_variable_assignment,
            "variable_declarator": CalculatedFromEdgeProcessor.__push_variable_assignment
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return []

    def get_edges_names(self) -> List[str]:
        return ["CALCULATED_FROM_EDGE", "TO_VARIABLE_ASSIGNMENT_EDGE"]

    @staticmethod
    def __push_variable_assignment(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index == 0:
            variable_node = get_child_of_type(node.children[0], "variable_identifier")
            meta.append(variable_node)

    @staticmethod
    def __pop_variable_assignment(node: AstNode, graph: GatedGraph, meta: Any):
        if node.children:
            meta.pop()

    @staticmethod
    def __add_calculated_from_edge(node: AstNode, graph: GatedGraph, meta: Any):
        for variable_assignment in meta:
            if node is variable_assignment or variable_assignment is None:
                continue
            graph.add_edge(variable_assignment, node, "CALCULATED_FROM_EDGE")
            graph.add_edge(node, variable_assignment, "TO_VARIABLE_ASSIGNMENT_EDGE")


class LastLexicalUseEdgeProcessor(EdgeProcessor):

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "method_declaration": LastLexicalUseEdgeProcessor.__enter_scope,
            "class_declaration": LastLexicalUseEdgeProcessor.__enter_scope,
            "block": LastLexicalUseEdgeProcessor.__enter_scope,
            "switch_statement": LastLexicalUseEdgeProcessor.__enter_scope,
            "if_statement": LastLexicalUseEdgeProcessor.__enter_scope,
            "for_statement": LastLexicalUseEdgeProcessor.__enter_scope,
            "for_range_loop": LastLexicalUseEdgeProcessor.__enter_scope,
            "while_statement": LastLexicalUseEdgeProcessor.__enter_scope,
            "do_statement": LastLexicalUseEdgeProcessor.__enter_scope
        }

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "variable_identifier": LastLexicalUseEdgeProcessor.__add_last_lexical_use_edge,
            "function_definition": LastLexicalUseEdgeProcessor.__exit_scope,
            "class_specifier": LastLexicalUseEdgeProcessor.__exit_scope,
            "compound_statement": LastLexicalUseEdgeProcessor.__exit_scope,
            "switch_statement": LastLexicalUseEdgeProcessor.__exit_scope,
            "if_statement": LastLexicalUseEdgeProcessor.__exit_scope,
            "for_statement": LastLexicalUseEdgeProcessor.__exit_scope,
            "for_range_loop": LastLexicalUseEdgeProcessor.__exit_scope,
            "while_statement": LastLexicalUseEdgeProcessor.__exit_scope,
            "do_statement": LastLexicalUseEdgeProcessor.__exit_scope
        }

    def get_enter_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "local_variable_declaration": LastLexicalUseEdgeProcessor.__report_variable_declaration,
            "formal_parameter": LastLexicalUseEdgeProcessor.__report_variable_declaration,
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return ScopedDictionary()

    def get_edges_names(self) -> List[str]:
        return ["NEXT_LEXICAL_USE_EDGE", "PREV_LEXICAL_USE_EDGE"]

    @staticmethod
    def __enter_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.enter_scope()

    @staticmethod
    def __exit_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.exit_scope()

    @staticmethod
    def __report_variable_declaration(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index < 1:
            return
        variable_identifier = get_child_of_type(node.children[child_index], "variable_identifier")
        if variable_identifier is not None:
            meta[variable_identifier.token] = NodeHolder(variable_identifier)

    @staticmethod
    def __add_last_lexical_use_edge(node: AstNode, graph: GatedGraph, meta: Any):
        last_use = meta[node.token]
        if last_use is None:
            return
        if last_use.node and last_use.node is not node:
            graph.add_edge(last_use.node, node, "NEXT_LEXICAL_USE_EDGE")
            graph.add_edge(node, last_use.node, "PREV_LEXICAL_USE_EDGE")
        last_use.node = node


class LastReadWriteEdgeProcessor(EdgeProcessor):

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "local_variable_declaration": LastReadWriteEdgeProcessor.__report_variable_declaration,
            "formal_parameter": LastReadWriteEdgeProcessor.__report_variable_declaration,

            "variable_declarator": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "plus_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "minus_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "multiply_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "divide_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "remainder_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "xor_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "and_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,
            "or_assignment_expression": LastReadWriteEdgeProcessor.__report_variable_assignment,

            "program": LastReadWriteEdgeProcessor.__enter_isolated_scope,
            "method_declaration": LastReadWriteEdgeProcessor.__enter_isolated_scope,
            "class_declaration": LastReadWriteEdgeProcessor.__enter_isolated_scope,
            "do_statement": LastReadWriteEdgeProcessor.__enter_scope,
            "block": LastReadWriteEdgeProcessor.__enter_scope,
            "for_statement": LastReadWriteEdgeProcessor.__enter_loop,
            "for_range_loop": LastReadWriteEdgeProcessor.__enter_loop,
            "while_statement": LastReadWriteEdgeProcessor.__enter_loop,
            "if_statement": LastReadWriteEdgeProcessor.__enter_if,
            "switch_statement": LastReadWriteEdgeProcessor.__enter_switch,
            "switch_block": LastReadWriteEdgeProcessor.__enter_case,
        }

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "variable_identifier": LastReadWriteEdgeProcessor.__add_read_write_edges,

            "program": LastReadWriteEdgeProcessor.__exit_isolated_scope,
            "method_declaration": LastReadWriteEdgeProcessor.__exit_isolated_scope,
            "class_declaration": LastReadWriteEdgeProcessor.__exit_isolated_scope,
            "do_statement": LastReadWriteEdgeProcessor.__exit_scope,
            "block": LastReadWriteEdgeProcessor.__exit_scope,
            "for_statement": LastReadWriteEdgeProcessor.__exit_loop,
            "for_range_loop": LastReadWriteEdgeProcessor.__exit_loop,
            "while_statement": LastReadWriteEdgeProcessor.__exit_loop,
            "if_statement": LastReadWriteEdgeProcessor.__exit_if,
            "switch_statement": LastReadWriteEdgeProcessor.__exit_switch,
            "switch_block": LastReadWriteEdgeProcessor.__exit_case,
        }

    def get_enter_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "if_statement": LastReadWriteEdgeProcessor.__enter_if_child,
            "for_statement": LastReadWriteEdgeProcessor.__enter_loop_child,
            "for_range_loop": LastReadWriteEdgeProcessor.__enter_loop_child,
            "while_statement": LastReadWriteEdgeProcessor.__enter_loop_child,
        }

    def get_exit_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "if_statement": LastReadWriteEdgeProcessor.__exit_if_child,
            "for_statement": LastReadWriteEdgeProcessor.__exit_loop_child,
            "for_range_loop": LastReadWriteEdgeProcessor.__exit_loop_child,
            "while_statement": LastReadWriteEdgeProcessor.__exit_loop_child,
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return LastReadWriteEdgeProcessor.Meta()

    def get_edges_names(self) -> List[str]:
        return ["LAST_READ_EDGE", "NEXT_READ_EDGE",
                "LAST_WRITE_EDGE", "NEXT_WRITE_EDGE"]

    @staticmethod
    def __enter_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.read_uses.enter_scope()
        meta.write_uses.enter_scope()

    @staticmethod
    def __exit_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.read_uses.merge_scopes([meta.read_uses.exit_scope()])
        meta.write_uses.merge_scopes([meta.write_uses.exit_scope()])

    @staticmethod
    def __enter_isolated_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.read_uses.enter_scope()
        meta.write_uses.enter_scope()

    @staticmethod
    def __exit_isolated_scope(node: AstNode, graph: GatedGraph, meta: Any):
        meta.read_uses.exit_scope()
        meta.write_uses.exit_scope()

    @staticmethod
    def __enter_if(node: AstNode, graph: GatedGraph, meta: Any):
        LastReadWriteEdgeProcessor.__enter_scope(node, graph, meta)
        meta.scopes_stack.append([])

    @staticmethod
    def __enter_if_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            meta.read_uses.enter_scope()
            meta.write_uses.enter_scope()

    @staticmethod
    def __exit_if_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            read_scope = meta.read_uses.exit_scope()
            write_scope = meta.write_uses.exit_scope()
            meta.scopes_stack[-1].append((read_scope, write_scope))

    @staticmethod
    def __exit_if(node: AstNode, graph: GatedGraph, meta: Any):
        children_scopes = meta.scopes_stack.pop()
        if len(children_scopes) == 1:
            read_scope, write_scope = children_scopes[0]
            meta.read_uses.merge_optional_scopes([read_scope])
            meta.write_uses.merge_optional_scopes([write_scope])
        else:
            read_scopes = [read_scope for read_scope, _ in children_scopes]
            meta.read_uses.merge_alternative_scopes(read_scopes)
            write_scopes = [write_scope for _, write_scope in children_scopes]
            meta.write_uses.merge_alternative_scopes(write_scopes)
        LastReadWriteEdgeProcessor.__exit_scope(node, graph, meta)

    @staticmethod
    def __enter_loop(node: AstNode, graph: GatedGraph, meta: Any):
        LastReadWriteEdgeProcessor.__enter_scope(node, graph, meta)
        meta.scopes_stack.append([])

    @staticmethod
    def __enter_loop_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index == len(node.children) - 1:
            meta.read_uses.enter_scope()
            meta.write_uses.enter_scope()

    @staticmethod
    def __exit_loop_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index == len(node.children) - 1:
            read_scope = meta.read_uses.exit_scope()
            write_scope = meta.write_uses.exit_scope()
            meta.scopes_stack[-1].append((read_scope, write_scope))

    @staticmethod
    def __exit_loop(node: AstNode, graph: GatedGraph, meta: Any):
        children_scopes = meta.scopes_stack.pop()
        read_scopes = [read_scope for read_scope, _ in children_scopes]
        meta.read_uses.merge_optional_scopes(read_scopes)
        write_scopes = [write_scope for _, write_scope in children_scopes]
        meta.write_uses.merge_optional_scopes(write_scopes)
        LastReadWriteEdgeProcessor.__exit_scope(node, graph, meta)

    @staticmethod
    def __enter_switch(node: AstNode, graph: GatedGraph, meta: Any):
        LastReadWriteEdgeProcessor.__enter_scope(node, graph, meta)
        meta.scopes_stack.append([])

    @staticmethod
    def __exit_switch(node: AstNode, graph: GatedGraph, meta: Any):
        case_scopes = meta.scopes_stack.pop()
        read_scopes = [read_scope for read_scope, _ in case_scopes]
        meta.read_uses.merge_alternative_scopes(read_scopes)
        write_scopes = [write_scope for _, write_scope in case_scopes]
        meta.write_uses.merge_alternative_scopes(write_scopes)
        LastReadWriteEdgeProcessor.__exit_scope(node, graph, meta)

    @staticmethod
    def __enter_case(node: AstNode, graph: GatedGraph, meta: Any):
        meta.read_uses.enter_scope()
        meta.write_uses.enter_scope()

    @staticmethod
    def __exit_case(node: AstNode, graph: GatedGraph, meta: Any):
        read_scope = meta.read_uses.exit_scope()
        write_scope = meta.write_uses.exit_scope()
        meta.scopes_stack[-1].append((read_scope, write_scope))

    @staticmethod
    def __report_variable_declaration(node: AstNode, graph: GatedGraph, meta: Any):
        for child in node.children[1:]:
            variable_identifier = get_child_of_type(child, "variable_identifier")
            if variable_identifier is not None:
                meta.variable_declarations.add(variable_identifier)
                meta.read_uses.report_declaration(variable_identifier.token)
                meta.write_uses.report_declaration(variable_identifier.token)

    @staticmethod
    def __report_variable_assignment(node: AstNode, graph: GatedGraph, meta: Any):
        variable_node = get_child_of_type(node.children[0], "variable_identifier")
        meta.variable_assignments.add(variable_node)

    @staticmethod
    def __report_field_identifier(node: AstNode, graph: GatedGraph, meta: Any):
        if node in meta.variable_assignments:
            meta.write_uses.report_uses(node.token, [node])

    @staticmethod
    def __add_read_write_edges(node: AstNode, graph: GatedGraph, meta: Any):
        for read_use in meta.read_uses.get_uses(node.token):
            graph.add_edge(node, read_use, "LAST_READ_EDGE")
            graph.add_edge(read_use, node, "NEXT_READ_EDGE")
        for write_use in meta.write_uses.get_uses(node.token):
            graph.add_edge(node, write_use, "LAST_WRITE_EDGE")
            graph.add_edge(write_use, node, "NEXT_WRITE_EDGE")
        if node in meta.variable_assignments:
            meta.write_uses.report_uses(node.token, [node])
        elif node not in meta.variable_declarations:
            meta.read_uses.report_uses(node.token, [node])

    class Meta:
        def __init__(self):
            self.read_uses = UsesScopedDictionary()
            self.write_uses = UsesScopedDictionary()
            self.variable_assignments = set()
            self.variable_declarations = set()
            self.scopes_stack = []


class FormalArgumentEdgeProcessor(EdgeProcessor):

    def get_enter_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "class_declaration": FormalArgumentEdgeProcessor.__enter_class_declaration,
            "method_declaration": FormalArgumentEdgeProcessor.__register_function_declarator
        }

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "method_invocation": FormalArgumentEdgeProcessor.__add_formal_argument_edge,
            "class_declaration": FormalArgumentEdgeProcessor.__exit_class_declaration
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return FormalArgumentEdgeProcessor.Meta()

    def get_edges_names(self) -> List[str]:
        return ["TO_FORMAL_ARGUMENT_EDGE", "TO_ARGUMENT_VALUE_EDGE"]

    @staticmethod
    def __enter_class_declaration(node: AstNode, graph: GatedGraph, meta: Any):
        class_name = get_child_of_type(node.children[0], "identifier").token
        meta.class_definitions_stack.append(class_name)

    @staticmethod
    def __exit_class_declaration(node: AstNode, graph: GatedGraph, meta: Any):
        meta.class_definitions_stack.pop()

    @staticmethod
    def __register_function_declarator(node: AstNode, graph: GatedGraph, meta: Any):
        parameters_list = get_child_of_type(node, "formal_parameters").children
        name = str(get_child_of_type(node, "identifier").token) + "_" + str(len(parameters_list))
        if meta.class_definitions_stack:
            name = meta.class_definitions_stack[-1] + "." + name
        if name in meta.functions_arguments:
            return
        arguments = []
        for parameter in parameters_list:
            identifier_node = get_child_of_type(parameter.children[-1], "variable_identifier")
            if identifier_node is None:
                return
            arguments.append(identifier_node)
        meta.functions_arguments[name] = arguments


    @staticmethod
    def __add_formal_argument_edge(node: AstNode, graph: GatedGraph, meta: Any):
        argument_list = node.children[-1].children
        if node.children[0].node_type == 'variable_identifier':
            variable_type = node.children[0].var_type
            field_identifier = get_child_of_type(node, "identifier").token
            if variable_type:
                name = variable_type + "." + field_identifier
            else:
                name = "?." + field_identifier
        else:
            identifier_node = get_child_of_type(node, "identifier")
            if identifier_node is None:
                return
            name = identifier_node.token
        name += "_" + str(len(argument_list))
        if name in meta.functions_arguments:
            for argument, value in zip(meta.functions_arguments[name], argument_list):
                graph.add_edge(value, argument, "TO_FORMAL_ARGUMENT_EDGE")
                graph.add_edge(argument, value, "TO_ARGUMENT_VALUE_EDGE")

    class Meta:
        def __init__(self):
            self.class_definitions_stack = []
            self.functions_arguments = dict()


class GuardedByEdgeProcessor(EdgeProcessor):

    def get_exit_node_callbacks(self) -> Dict[str, Callable[[AstNode, GatedGraph, Any], None]]:
        return {
            "variable_identifier": GuardedByEdgeProcessor.__add_guarded_by_edges
        }

    def get_enter_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "if_statement": GuardedByEdgeProcessor.__enter_if_child,
            "for_statement": GuardedByEdgeProcessor.__enter_for_child,
            "while_statement": GuardedByEdgeProcessor.__enter_while_child,
        }

    def get_exit_child_callbacks(self) -> Dict[str, Callable[[AstNode, int, GatedGraph, Any], None]]:
        return {
            "if_statement": GuardedByEdgeProcessor.__exit_if_child,
            "for_statement": GuardedByEdgeProcessor.__exit_for_child,
            "while_statement": GuardedByEdgeProcessor.__exit_while_child,
        }

    def init_meta(self, root: AstNode, graph: GatedGraph) -> Any:
        return []

    def get_edges_names(self) -> List[str]:
        return ["GUARDED_BY_EDGE", "GUARD_TO_EDGE",
                "NEGATIVE_GUARDED_BY_EDGE", "NEGATIVE_GUARD_TO_EDGE"]

    @staticmethod
    def __enter_if_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            meta.append(GuardedByEdgeProcessor.Guard(node.children[0].children[0], child_index == 2))

    @staticmethod
    def __exit_if_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            meta.pop()

    @staticmethod
    def __enter_while_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            meta.append(GuardedByEdgeProcessor.Guard(node.children[0].children[0]))

    @staticmethod
    def __exit_while_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        if child_index != 0:
            meta.pop()

    @staticmethod
    def __enter_for_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        children_num = len(node.children)
        if child_index >= children_num - 2 and children_num == 4:
            meta.append(GuardedByEdgeProcessor.Guard(node.children[1]))

    @staticmethod
    def __exit_for_child(node: AstNode, child_index: int, graph: GatedGraph, meta: Any):
        children_num = len(node.children)
        if child_index >= children_num - 2 and children_num == 4:
            meta.pop()

    @staticmethod
    def __add_guarded_by_edges(node: AstNode, graph: GatedGraph, meta: Any):
        var_name = node.token
        for guard in meta:
            if var_name in guard.used_variables:
                if guard.negative_branch:
                    graph.add_edge(node, guard.guard_node, "NEGATIVE_GUARDED_BY_EDGE")
                    graph.add_edge(guard.guard_node, node, "NEGATIVE_GUARD_TO_EDGE")
                else:
                    graph.add_edge(node, guard.guard_node, "GUARDED_BY_EDGE")
                    graph.add_edge(guard.guard_node, node, "GUARD_TO_EDGE")

    class Guard:
        def __init__(self, guard_node: AstNode, negative_branch=False):
            self.guard_node = guard_node
            self.used_variables = get_variables_names(guard_node) if guard_node is not None else set()
            self.negative_branch = negative_branch


class NodeHolder:
    def __init__(self, node=None):
        self.node = node


def get_child_of_type(root: AstNode, node_type: str):
    if root.node_type == node_type:
        return root
    for child in root.children:
        tmp = get_child_of_type(child, node_type)
        if tmp:
            return tmp
    return None


def get_variables_names(root: AstNode):
    buffer = set()

    def dfs(node: AstNode):
        if node.node_type == 'variable_identifier':
            buffer.add(node.token)
        for child in node.children:
            dfs(child)

    dfs(root)
    return buffer


def apply_for_each_pair(function, pairs):
    for first, second in pairs:
        function(first, second)
