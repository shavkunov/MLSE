from tree_sitter import Language
import os

Language.build_library(
    # Store the library in the `build` directory
    'build/languages.so',

    # Include one or more languages
    [
        os.path.join(os.path.dirname(__file__), 'vendor', 'tree-sitter-java')
    ]
)