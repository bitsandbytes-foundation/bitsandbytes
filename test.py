#!/usr/bin/env python3
"""
Recursive script to extract all classes, functions, and methods from Python files
with support for both grep-based and AST-based extraction.
"""

import os
import re
import ast
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any


class GrepExtractor:
    """Grep-based code extraction"""
    
    @staticmethod
    def get_indent_level(line: str) -> int:
        """Calculate the indentation level of a line"""
        count = 0
        for char in line:
            if char == ' ':
                count += 1
            elif char == '\t':
                count += 4  # Assuming 1 tab = 4 spaces
            else:
                break
        return count
    
    @staticmethod
    def find_definitions_with_grep(file_path: str) -> List[Tuple[int, str]]:
        """Use grep to find all class and function definitions with line numbers"""
        try:
            # Pattern to match class and function definitions
            pattern = r'^[ \t]*(class|def|async[ \t]+def)[ \t]+[a-zA-Z_][a-zA-Z0-9_]*[ \t]*\('
            
            # Run grep command
            result = subprocess.run(
                ['grep', '-n', '-E', pattern, file_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return []
            
            # Parse grep output
            definitions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        line_num = int(parts[0])
                        content = parts[1]
                        definitions.append((line_num, content))
            
            return definitions
        except Exception:
            return []
    
    @staticmethod
    def extract_signature(line: str) -> Optional[str]:
        """Extract the function/class signature from a line"""
        # Remove leading whitespace
        line = line.strip()
        
        # Match class statement
        class_match = re.match(r'class\s+(\w+)\s*(\([^)]*\))?', line)
        if class_match:
            name = class_match.group(1)
            params = class_match.group(2) if class_match.group(2) else '()'
            return f"{name}{params}"
        
        # Match function statement (including async)
        def_match = re.match(r'(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', line)
        if def_match:
            name = def_match.group(2)
            params = def_match.group(3).strip()
            # Extract parameter names only
            param_list = []
            for param in params.split(','):
                param = param.strip()
                if param:
                    # Extract just the parameter name
                    param_name = re.match(r'(\*{0,2}\w+)', param)
                    if param_name:
                        param_list.append(param_name.group(1))
            
            return f"{name}({', '.join(param_list)})"
        
        return None
    
    @staticmethod
    def find_end_line(lines: List[str], start_idx: int, start_indent: int) -> int:
        """Find the end line of a code block"""
        if start_idx >= len(lines) - 1:
            return len(lines)
        
        i = start_idx + 1
        in_docstring = False
        docstring_quotes = None
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Handle docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_quotes = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(docstring_quotes) == 1:
                    in_docstring = True
                i += 1
                continue
            
            if in_docstring:
                if docstring_quotes in stripped:
                    in_docstring = False
                i += 1
                continue
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                i += 1
                continue
            
            # Check indentation
            current_indent = GrepExtractor.get_indent_level(line)
            if current_indent <= start_indent and stripped:
                return i
            
            i += 1
        
        return len(lines)
    
    @staticmethod
    def process_file(file_path: str) -> List[Dict[str, Any]]:
        """Process a single Python file using grep"""
        elements = []
        
        # Get all definitions using grep
        definitions = GrepExtractor.find_definitions_with_grep(file_path)
        if not definitions:
            return elements
        
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return elements
        
        # Process each definition
        for line_num, content in definitions:
            indent_level = GrepExtractor.get_indent_level(content)
            signature = GrepExtractor.extract_signature(content)
            
            if signature:
                # Find the end line
                end_line = GrepExtractor.find_end_line(lines, line_num - 1, indent_level)
                
                # Extract the name from signature
                name = signature.split('(')[0]
                
                elements.append({
                    'name': name,
                    'signature': signature,
                    'start_line': line_num,
                    'end_line': end_line,
                    'type': 'function' if 'def' in content else 'class'
                })
        
        return elements


class ASTExtractor:
    """AST-based code extraction"""
    
    class CodeVisitor(ast.NodeVisitor):
        """AST visitor to extract function and class definitions"""
        
        def __init__(self):
            self.elements = []
            
        def visit_ClassDef(self, node: ast.ClassDef):
            """Visit class definitions"""
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            
            signature = f"{node.name}({', '.join(bases)})" if bases else f"{node.name}()"
            
            self.elements.append({
                'name': node.name,
                'signature': signature,
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'type': 'class'
            })
            
            self.generic_visit(node)
        
        def visit_FunctionDef(self, node: ast.FunctionDef):
            """Visit function definitions"""
            self._process_function(node)
            
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            """Visit async function definitions"""
            self._process_function(node, is_async=True)
        
        def _process_function(self, node: Any, is_async: bool = False):
            """Process function or async function"""
            params = []
            
            # Add positional arguments
            for arg in node.args.args:
                params.append(arg.arg)
            
            # Add *args if present
            if node.args.vararg:
                params.append(f"*{node.args.vararg.arg}")
            
            # Add keyword-only arguments
            for arg in node.args.kwonlyargs:
                params.append(arg.arg)
            
            # Add **kwargs if present
            if node.args.kwarg:
                params.append(f"**{node.args.kwarg.arg}")
            
            signature = f"{node.name}({', '.join(params)})"
            
            self.elements.append({
                'name': node.name,
                'signature': signature,
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'type': 'function'
            })
            
            self.generic_visit(node)
    
    @staticmethod
    def process_file(file_path: str) -> List[Dict[str, Any]]:
        """Process a Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse the AST
            tree = ast.parse(source, filename=file_path)
            
            # Extract elements
            visitor = ASTExtractor.CodeVisitor()
            visitor.visit(tree)
            
            # Sort by line number
            visitor.elements.sort(key=lambda x: x['start_line'])
            
            return visitor.elements
            
        except (SyntaxError, Exception):
            return []


def find_python_files(root_path: Path, recursive: bool = True) -> List[Path]:
    """Find all Python files in directory"""
    if recursive:
        return list(root_path.rglob("*.py"))
    else:
        return list(root_path.glob("*.py"))


def format_output(file_path: Path, elements: List[Dict[str, Any]], root_path: Path) -> str:
    """Format the output with relative paths"""
    # Get relative path
    try:
        relative_path = file_path.relative_to(root_path)
    except ValueError:
        relative_path = file_path
    
    output = []
    output.append(str(relative_path))
    output.append("=" * len(str(relative_path)))
    
    for element in elements:
        start = element['start_line']
        end = element['end_line']
        limit = end - start + 1
        
        signature = element['signature']
        output.append(f"{signature} # Read(file_path={relative_path}, "
                     f"offset={start}, limit={limit})")
    
    return '\n'.join(output)


def benchmark_methods(file_path: Path) -> Dict[str, float]:
    """Benchmark both extraction methods"""
    results = {}
    
    # Benchmark grep method
    start_time = time.time()
    grep_elements = GrepExtractor.process_file(str(file_path))
    results['grep_time'] = time.time() - start_time
    results['grep_count'] = len(grep_elements)
    
    # Benchmark AST method
    start_time = time.time()
    ast_elements = ASTExtractor.process_file(str(file_path))
    results['ast_time'] = time.time() - start_time
    results['ast_count'] = len(ast_elements)
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract Python code structure')
    parser.add_argument('path', nargs='?', default='.', 
                       help='Path to Python file or directory (default: current directory)')
    parser.add_argument('--method', choices=['grep', 'ast'], default='grep',
                       help='Extraction method (default: grep)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Don\'t recursively search directories')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark both methods')
    parser.add_argument('--output', '-o', 
                       help='Output file (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Convert to Path object
    path = Path(args.path)
    root_path = path if path.is_dir() else path.parent
    
    # Find files to process
    if path.is_file() and path.suffix == '.py':
        files = [path]
    elif path.is_dir():
        files = find_python_files(path, recursive=not args.no_recursive)
    else:
        print(f"Error: {path} is not a Python file or directory", file=sys.stderr)
        sys.exit(1)
    
    if not files:
        print("No Python files found")
        sys.exit(0)
    
    # Benchmark mode
    if args.benchmark:
        print("Benchmarking both methods...\n")
        total_grep_time = 0
        total_ast_time = 0
        
        for file_path in files[:10]:  # Benchmark first 10 files
            results = benchmark_methods(file_path)
            print(f"{file_path.relative_to(root_path)}:")
            print(f"  Grep: {results['grep_time']:.4f}s ({results['grep_count']} items)")
            print(f"  AST:  {results['ast_time']:.4f}s ({results['ast_count']} items)")
            total_grep_time += results['grep_time']
            total_ast_time += results['ast_time']
        
        print(f"\nTotal grep time: {total_grep_time:.4f}s")
        print(f"Total AST time:  {total_ast_time:.4f}s")
        print(f"Grep is {total_ast_time/total_grep_time:.1f}x faster")
        return
    
    # Process files
    all_output = []
    extractor = GrepExtractor if args.method == 'grep' else ASTExtractor
    
    for file_path in sorted(files):
        elements = extractor.process_file(str(file_path))
        
        if elements:
            output = format_output(file_path, elements, root_path)
            all_output.append(output)
    
    # Output results
    final_output = '\n\n'.join(all_output)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"Output saved to: {args.output}")
    else:
        print(final_output)


if __name__ == "__main__":
    main()
