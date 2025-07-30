import ast
import networkx as nx
import random
from pathlib import Path
import sys
import argparse
from typing import Dict, Set, List, Tuple, Optional


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_function = None
        self.objects_used = {}  # Track objects used in each function
        self.function_locations = {}  # Track line numbers

    def visit_FunctionDef(self, node):
        function_name = node.name
        self.graph.add_node(function_name)
        self.function_locations[function_name] = node.lineno

        # Store previous function to handle nesting
        previous_function = self.current_function
        self.current_function = function_name

        # Visit all children in the function body
        self.objects_used[function_name] = set()
        self.generic_visit(node)

        # Restore previous function context
        self.current_function = previous_function

    def visit_Call(self, node):
        if self.current_function:
            called_function = None
            line_num = getattr(node, 'lineno', None)

            # Handle different types of function calls
            if isinstance(node.func, ast.Name):
                called_function = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # For method calls like obj.method()
                obj_name = self._get_attribute_source(node.func)
                method_name = node.func.attr
                called_function = f"{obj_name}.{method_name}"

                # Track line number for method calls
                if line_num and called_function not in self.function_locations:
                    self.function_locations[called_function] = line_num

            if called_function:
                self.graph.add_node(called_function)
                self.graph.add_edge(self.current_function, called_function)

        # Visit arguments to find more function calls
        self.generic_visit(node)

    def _get_attribute_source(self, node):
        """Helper to get the source of an attribute (e.g., 'obj' from 'obj.method')"""
        if isinstance(node.value, ast.Name):
            return node.value.id
        elif isinstance(node.value, ast.Attribute):
            # Handle nested attributes like a.b.method()
            return self._get_attribute_source(node.value) + "." + node.value.attr
        elif isinstance(node.value, ast.Call):
            # Handle method calls on function returns like func().method()
            return "(result)"
        return "object"

    def visit_Name(self, node):
        # Track objects/variables used in function
        if self.current_function and isinstance(node.ctx, ast.Load):
            self.objects_used[self.current_function].add(node.id)
        self.generic_visit(node)


def build_call_graph(file_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[str]], Dict[str, int]]:
    """Build a call graph from a Python file."""
    with open(file_path, 'r') as file:
        source_code = file.read()

    # Parse the AST
    tree = ast.parse(source_code)

    # Visit the AST and build the call graph
    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    # Find functions that are called but not defined in this file
    # (likely imported functions or methods on objects)
    all_called = set()
    for _, successors in nx.bfs_successors(visitor.graph, list(visitor.graph.nodes())[0] if visitor.graph.nodes() else None):
        all_called.update(successors)

    defined_funcs = set(visitor.function_locations.keys())
    external_funcs = all_called - defined_funcs

    print(f"External function calls: {len(external_funcs)}")

    return visitor.graph, visitor.objects_used, visitor.function_locations


def generate_random_trace(graph: nx.DiGraph, min_depth: int = 3, max_depth: int = 10) -> List[str]:
    """Generate a random trace through the call graph with a minimum depth when possible."""
    if not graph.nodes():
        return []

    # Find nodes that have outgoing edges as potential starting points
    starting_candidates = [n for n in graph.nodes() if graph.out_degree(n) > 0]
    if not starting_candidates:
        starting_candidates = list(graph.nodes())

    current = random.choice(starting_candidates)
    trace = [current]
    visited = set([current])

    # Follow a random path down the graph
    depth_attempts = 0
    while depth_attempts < 50:  # Allow more attempts for deeper traces
        # Get successors that haven't been visited to avoid cycles
        successors = [s for s in graph.successors(current) if s not in visited]

        if not successors:
            # If we're at a leaf node but haven't reached min_depth, try backtracking
            if len(trace) < min_depth and len(trace) > 1:
                # Remove the current dead-end
                visited.remove(current)
                trace.pop()
                current = trace[-1]
                continue
            # Otherwise, we've reached a valid end point
            break

        # Choose a successor with preference for those that have their own successors
        weighted_successors = []
        for s in successors:
            # Assign weight based on number of outgoing edges
            weight = max(1, graph.out_degree(s))
            weighted_successors.extend([s] * weight)

        if weighted_successors:
            current = random.choice(weighted_successors)
        else:
            current = random.choice(successors)

        trace.append(current)
        visited.add(current)

        # Stop if we've reached desired max depth
        if len(trace) >= max_depth:
            break

        depth_attempts += 1

    return trace


def print_stack_trace(trace: List[str], file_path: str, objects_used: Dict[str, Set[str]],
                     function_locations: Dict[str, int]):
    """Print a stack-trace-like representation of the function call path."""
    if not trace:
        print("No functions found in the trace.")
        return

    file_name = Path(file_path).name
    print(f"\n{'=' * 60}")
    print(f"RANDOM FUNCTION CALL TRACE IN: {file_name}")
    print(f"{'=' * 60}")

    for i, func in enumerate(trace):
        line_num = function_locations.get(func, '?')
        indent = '  ' * i

        # For the function name display
        if i < len(trace) - 1:
            arrow = "↓ calls"
        else:
            arrow = "⊥ (end)"

        # Handle method calls differently
        if '.' in func:
            obj_name, method_name = func.rsplit('.', 1)
            print(f"{indent}File \"{file_name}\", line {line_num}, in {obj_name} object")
            print(f"{indent}  Method call: {method_name}()")
        else:
            print(f"{indent}File \"{file_name}\", line {line_num}, in {func}()")

        # Show objects used in this function
        if func in objects_used and objects_used[func]:
            obj_list = ", ".join(objects_used[func])
            print(f"{indent}  [Objects used: {obj_list}]")

        # Show the arrow for the next function call
        if i < len(trace) - 1:
            print(f"{indent}  {arrow}")


def main():
    parser = argparse.ArgumentParser(description="Generate a random function call trace from Python code")
    parser.add_argument("file", help="Python file to analyze")
    parser.add_argument("--traces", type=int, default=1, help="Number of random traces to generate")
    parser.add_argument("--min-depth", type=int, default=3, help="Minimum depth of trace to try to achieve")
    parser.add_argument("--max-depth", type=int, default=15, help="Maximum depth of trace")
    parser.add_argument("--max-attempts", type=int, default=100, help="Maximum number of attempts to find a trace that meets min-depth")
    args = parser.parse_args()

    try:
        graph, objects_used, function_locations = build_call_graph(args.file)

        if not graph.nodes():
            print(f"No functions found in {args.file}")
            return

        # Calculate graph stats
        total_functions = len(graph.nodes())
        total_calls = len(graph.edges())
        max_call_depth = nx.dag_longest_path_length(nx.DiGraph(graph)) if nx.is_directed_acyclic_graph(graph) else "unknown (contains cycles)"

        print(f"Found {total_functions} functions with {total_calls} call relationships in {args.file}")
        print(f"Maximum theoretical call depth: {max_call_depth}")
        print(f"Generating {args.traces} traces with minimum depth {args.min_depth}...")

        traces_generated = 0
        attempts = 0

        while traces_generated < args.traces and attempts < args.max_attempts:
            trace = generate_random_trace(graph, args.min_depth, args.max_depth)
            attempts += 1

            if len(trace) >= args.min_depth:
                print_stack_trace(trace, args.file, objects_used, function_locations)
                print(f"Trace depth: {len(trace)} (found after {attempts} attempts)")
                traces_generated += 1
                attempts = 0  # Reset attempts counter for next trace

        if traces_generated < args.traces:
            print(f"\nWARNING: Could only generate {traces_generated} traces of minimum depth {args.min_depth} "
                  f"after {args.max_attempts} attempts.")
            print(f"The codebase may not have enough deep call chains to satisfy the requested minimum depth.")
            # Optionally suggest a smaller depth
            if args.min_depth > 2:
                print(f"Try using a smaller --min-depth value (e.g., {args.min_depth - 1}).")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
