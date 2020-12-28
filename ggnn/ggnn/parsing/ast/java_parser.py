from itertools import count

from .common import AstNode, AstParser, ScopedDictionary


class JavaParserMeta:
    def __init__(self, content):
        self.content = content
        self.id_generator = count(0, 1)
        self.node_id_stack = []
        self.node_type_stack = []
        self.node_token_stack = []
        self.var_type_stack = []
        self.var_dictionary = ScopedDictionary()

    def push(self, node_id, node_type, node_token, var_type):
        self.node_id_stack.append(node_id)
        self.node_type_stack.append(node_type)
        self.node_token_stack.append(node_token)
        self.var_type_stack.append(var_type)

    def peek(self):
        return self.node_id_stack[-1], self.node_type_stack[-1], \
               self.node_token_stack[-1], self.var_type_stack[-1]

    def pop(self):
        return self.node_id_stack.pop(), self.node_type_stack.pop(), \
               self.node_token_stack.pop(), self.var_type_stack.pop()


class JavaParser(AstParser):
    def __init__(self):
        super().__init__("java")
        self.ignored_types = {"escape_sequence", "comment", "ERROR"}
        self.scoped_types = {
            "function_definition", "class_specifier",
            "switch_statement", "if_statement",
            "for_statement", "for_range_loop",
            "while_statement", "do_statement",
            "compound_statement"
        }
        self.data_extractors = {
            "identifier": self.__get_identifier_data,
            "type_identifier": self.__get_default_labeled_node_data,
            "field_identifier": self.__get_identifier_data,
            "sized_type_specifier": self.__get_default_labeled_node_data,
            "namespace_identifier": self.__get_default_labeled_node_data,
            "number_literal": self.__get_default_labeled_node_data,
            "boolean_type": self.__get_default_labeled_node_data,
            "dimensions": self.__get_default_labeled_node_data,
            "decimal_integer_literal": self.__get_default_labeled_node_data,
            "decimal_floating_point_literal": self.__get_default_labeled_node_data,
            "floating_point_type": self.__get_default_labeled_node_data,
            "access_specifier": self.__get_default_labeled_node_data,
            "type_qualifier": self.__get_default_labeled_node_data,
            "char_literal": self.__get_string_literal_data,
            "string_literal": self.__get_string_literal_data,
            "assignment_expression": self.__get_assignment_data,
            "binary_expression": self.__get_binary_expression_data,
            "unary_expression": self.__get_unary_expression_data,
            "pointer_expression": self.__get_pointer_expression_data,
            "update_expression": self.__get_update_expression_data,
        }
        self.type_triggers = {
            "local_variable_declaration": [self.__process_declaration],
            "formal_parameter": [self.__process_declaration],
        }
        self.assignments_names = {
            "=": "assignment_expression",
            "+=": "plus_assignment_expression",
            "-=": "minus_assignment_expression",
            "*=": "multiply_assignment_expression",
            "/=": "divide_assignment_expression",
            "%=": "remainder_assignment_expression",
            "^=": "xor_assignment_expression",
            "&=": "and_assignment_expression",
            "|=": "or_assignment_expression",
            ">>=": "right_shift_assignment_expression",
            "<<=": "right_shift_assignment_expression",

            "ERROR": "unexpected_assignment_expression"
        }
        self.binary_operators_names = {
            "+": "plus_binary_expression",
            "-": "minus_binary_expression",
            "*": "multiply_binary_expression",
            "/": "divide_binary_expression",
            "%": "remainder_binary_expression",
            "<<": "left_shift_binary_expression",
            ">>": "right_shift_binary_expression",
            "^": "xor_binary_expression",
            "&": "bit_and_binary_expression",
            "|": "bit_or_binary_expression",
            "&&": "logical_and_binary_expression",
            "||": "logical_or_binary_expression",

            "<": "less_relational_binary_expression",
            ">": "greater_relational_binary_expression",
            "<=": "less_equal_relational_binary_expression",
            ">=": "greater_equal_relational_binary_expression",
            "==": "equal_relational_binary_expression",
            "!=": "not_equal_relational_binary_expression",

            "ERROR": "unknown_binary_expression"
        }
        self.unary_operators_names = {
            "+": "plus_unary_expression",
            "-": "minus_unary_expression",
            "~": "complement_unary_expression",
            "!": "logical_negation_unary_expression",

            "ERROR": "unexpected_unary_expression"
        }
        self.pointer_operators_names = {
            "*": "indirection_pointer_expression",
            "&": "address_of_pointer_expression",

            "ERROR": "unexpected_pointer_expression"
        }
        self.update_prefix_operators_names = {
            "++": "prefix_increment_expression",
            "--": "prefix_decrement_expression",

            "ERROR": "unexpected_prefix_expression"
        }
        self.update_postfix_operators_names = {
            "++": "postfix_increment_expression",
            "--": "postfix_decrement_expression",

            "ERROR": "unexpected_postfix_expression"
        }
        self.var_types_nodes = {
            "generic_type",
            "type_identifier",
            "type_arguments",
            "integral_type",
            "floating_point_type",
            "boolean_type",
            "array_type"
        }

    def create_node(self, cursor, children, meta: JavaParserMeta):
        node_id, node_type, node_token, var_type = meta.peek()
        if node_id is None:
            return None
        return AstNode(node_id, node_type, node_token, children, var_type)

    def enter_node(self, cursor, meta: JavaParserMeta):
        raw_node = cursor.node
        if not raw_node.is_named or raw_node.type in self.ignored_types:
            meta.push(None, None, None, None)
            return
        data_extractor = self.data_extractors.get(raw_node.type, self.__get_default_node_data)
        node_id, node_type, node_token, var_type = data_extractor(raw_node, meta)
        meta.push(node_id, node_type, node_token, var_type)
        if node_type in self.scoped_types:
            meta.var_dictionary.enter_scope()
        triggers = self.type_triggers.get(node_type)
        if triggers:
            for trigger in triggers:
                trigger(raw_node, meta)

    def exit_node(self, cursor, meta: JavaParserMeta):
        node_id, node_type, node_token, var_type = meta.pop()
        if node_type in self.scoped_types:
            meta.var_dictionary.exit_scope()

    def init_meta(self, ast, content):
        return JavaParserMeta(content)

    def __get_default_node_data(self, raw_node, meta):
        return next(meta.id_generator), raw_node.type, None, None

    def __get_default_labeled_node_data(self, raw_node, meta):
        return next(meta.id_generator), raw_node.type, meta.content[raw_node.start_byte:raw_node.end_byte], None

    def __get_string_literal_data(self, raw_node, meta):
        return next(meta.id_generator), raw_node.type, meta.content[raw_node.start_byte + 1:raw_node.end_byte - 1], None

    def __get_assignment_data(self, raw_node, meta):
        return next(meta.id_generator), self.assignments_names[raw_node.children[1].type], None, None

    def __get_binary_expression_data(self, raw_node, meta):
        return next(meta.id_generator), self.binary_operators_names[raw_node.children[1].type], None, None

    def __get_unary_expression_data(self, raw_node, meta):
        return next(meta.id_generator), self.unary_operators_names[raw_node.children[0].type], None, None

    def __get_pointer_expression_data(self, raw_node, meta):
        return next(meta.id_generator), self.pointer_operators_names[raw_node.children[0].type], None, None

    def __get_update_expression_data(self, raw_node, meta):
        raw_children = raw_node.children
        if raw_children[0].type in self.update_prefix_operators_names:
            return next(meta.id_generator), self.update_prefix_operators_names[raw_children[0].type], None, None
        else:
            return next(meta.id_generator), self.update_postfix_operators_names[raw_children[1].type], None, None

    def __get_identifier_data(self, raw_node, meta):
        node_token = meta.content[raw_node.start_byte:raw_node.end_byte]
        var_type = meta.var_dictionary[node_token]
        node_type = 'variable_identifier' if var_type and raw_node.type == 'identifier' else raw_node.type
        return next(meta.id_generator), node_type, node_token, var_type

    def __process_declaration(self, raw_node, meta):
        children = raw_node.children
        pos, var_type = get_node_of_type(children, self.var_types_nodes)
        if var_type is None:
            return
        var_type = self.__get_type_name(var_type, meta.content)
        for child in children[pos:]:
            JavaParser.__update_types(var_type, child, meta)

    @staticmethod
    def __update_types(base_type, raw_node, meta):
        if raw_node.type == 'identifier' or raw_node.type == 'variable_identifier':
            var_name = meta.content[raw_node.start_byte:raw_node.end_byte]
            meta.var_dictionary[var_name] = base_type
        elif raw_node.type == 'variable_declarator':
            JavaParser.__update_types(base_type, raw_node.children[0], meta)

    @staticmethod
    def __get_type_name(raw_node, content):
        if raw_node.type == 'type_identifier' or raw_node.type == 'floating_point_type' \
                or raw_node.type == 'boolean_type' or raw_node.type == 'integral_type':
            return content[raw_node.start_byte:raw_node.end_byte]
        if raw_node.type == 'generic_type':
            children = raw_node.children
            template_type = "".join([JavaParser.__get_type_name(child, content) for child in children[1].children])
            return JavaParser.__get_type_name(children[0], content) + template_type
        if raw_node.type == 'array_type':
            children = raw_node.children
            return JavaParser.__get_type_name(children[0], content) + content[
                                                                      children[1].start_byte:children[1].end_byte]
        return raw_node.type


def get_node_of_type(nodes, types):
    for pos, node in enumerate(nodes):
        if node.type in types:
            return pos, node
    return -1, None


def get_node_not_of_type(nodes, types):
    for pos, node in enumerate(nodes):
        if node.type not in types:
            return pos, node
    return -1, None
