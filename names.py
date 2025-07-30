import ast
from pathlib import Path


def show_info(functionNode, isclass=False):
    tab = '\t' if isclass else ''
    with open('./functions.txt', 'a+') as f:
        f.write(f"{tab}Function name: {functionNode.name}\n")
        f.write(f"{tab}Args:\n")
        for arg in functionNode.args.args:
            f.write(f"{tab}\t: {arg.arg}\n")


for path in Path('./bitsandbytes').rglob('*.py'):
    print(path)
    with open(path) as file:
        node = ast.parse(file.read())

    functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

    for function in functions:
        show_info(function)

    for class_ in classes:
        with open('./functions.txt', 'a+') as f:
            f.write(f"Class name: {class_.name}\n")
        methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
        for method in methods:
            show_info(method, isclass=True)
