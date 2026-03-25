"""
Automatic LaTeX Tree Generator for Genetic Programming Equations

This module provides functions to automatically convert string representations 
of mathematical expressions (from GP systems) to LaTeX forest tree visualizations.

Author: Generated for GP-SDE project
"""

import sympy
import re
import ast
import pandas as pd
import os
from typing import Union, List


def parse_equation_string(equation_str: str) -> List[sympy.Expr]:
    """
    Parse equation string from CSV format to list of sympy expressions.
    
    Parameters:
    -----------
    equation_str : str
        String representation of equation(s), e.g., "[laplacian + u_x, 0.101*u]"
        
    Returns:
    --------
    List[sympy.Expr]
        List of parsed sympy expressions
    """
    # Remove outer brackets and split by comma while respecting nested structures
    equation_str = equation_str.strip()
    if equation_str.startswith('[') and equation_str.endswith(']'):
        equation_str = equation_str[1:-1]
    
    # Split by comma, being careful about nested expressions
    parts = re.split(r',(?![^()]*\))', equation_str)
    
    sympy_expressions = []
    for part in parts:
        part = part.strip()
        try:
            # Handle special variables
            part_processed = part.replace('laplacian', 'L')
            sympy_expressions.append(sympy.sympify(part_processed))
        except:
            # If sympify fails, try to parse as a numeric value
            try:
                sympy_expressions.append(sympy.sympify(float(part)))
            except:
                # Last resort - treat as symbol
                sympy_expressions.append(sympy.Symbol(part))
    
    return sympy_expressions


def sympy_to_latex_tree(expr: sympy.Expr, highlight_color: str = None) -> str:
    """
    Convert a sympy expression to LaTeX forest tree format.
    
    Parameters:
    -----------
    expr : sympy.Expr
        The sympy expression to convert
    highlight_color : str, optional
        Color to highlight the tree (e.g., 'red', 'blue', 'ForestGreen')
        
    Returns:
    --------
    str
        LaTeX forest tree representation  
    """
    
    def _add_color_attrs(node_content: str, has_children: bool = False) -> str:
        """Add color attributes to a node if highlight_color is specified."""
        if not highlight_color:
            return node_content
        
        # Find the closing $ of the node content
        dollar_pos = node_content.find('$')
        if dollar_pos != -1:
            second_dollar = node_content.find('$', dollar_pos + 1)
            if second_dollar != -1:
                insert_pos = second_dollar + 1
                if has_children:
                    # For nodes with children, define color for text and children
                    color_attrs = f", text={{{highlight_color}}}, for children={{edge={{{highlight_color}}}, text={{{highlight_color}}}}}"
                else:
                    # For leaf nodes, only define text color
                    color_attrs = f", text={{{highlight_color}}}"
                
                return node_content[:insert_pos] + color_attrs + node_content[insert_pos:]
        return node_content
    
    def _expr_to_tree(expr):
        # Handle constants/numbers
        if expr.is_number:
            # Round to 3 decimal places for cleaner display
            if abs(float(expr)) < 0.001 and float(expr) != 0:
                formatted_num = f"{float(expr):.3e}"
            else:
                formatted_num = f"{float(expr):.3f}".rstrip('0').rstrip('.')
            return _add_color_attrs(f"[${formatted_num}$]", has_children=False)
        
        # Handle symbols/variables
        if expr.is_symbol:
            var_name = str(expr)
            # Convert common variable names to proper LaTeX
            latex_vars = {
                'x0': 'x_0', 'x1': 'x_1', 'x2': 'x_2', 'x3': 'x_3',
                'u': 'u', 'v': 'v', 'w': 'w',
                'u_x': 'u_x', 'u_y': 'u_y',
                'u_xx': 'u_xx'
            }
            latex_name = latex_vars.get(var_name, var_name)
            return _add_color_attrs(f"[${latex_name}$]", has_children=False)
        
        # Handle function calls (sin, cos, etc.)
        if hasattr(expr.func, '__name__') and expr.func.__name__ in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            func_name = f"\\{expr.func.__name__}"
            arg_tree = _expr_to_tree(expr.args[0])
            node_content = f"[${func_name}$\n    {arg_tree}\n]"
            return _add_color_attrs(node_content, has_children=True)
        
        # Handle binary operations
        if len(expr.args) == 2:
            left, right = expr.args
            left_tree = _expr_to_tree(left)
            right_tree = _expr_to_tree(right)
            
            if expr.func == sympy.Add:
                node_content = f"[$+$\n    {left_tree}\n    {right_tree}\n]"
                return _add_color_attrs(node_content, has_children=True)
            elif expr.func == sympy.Mul:
                node_content = f"[$\\times$\n    {left_tree}\n    {right_tree}\n]"
                return _add_color_attrs(node_content, has_children=True)
            elif expr.func == sympy.Pow:
                node_content = f"[$^{{\\wedge}}$\n    {left_tree}\n    {right_tree}\n]"
                return _add_color_attrs(node_content, has_children=True)
        
        # Handle subtraction (check for negative terms in addition)
        if expr.func == sympy.Add:
            # Separate positive and negative terms
            pos_terms = []
            neg_terms = []
            
            for arg in expr.args:
                if (arg.is_number and float(arg) < 0) or (hasattr(arg, 'could_extract_minus_sign') and arg.could_extract_minus_sign()):
                    neg_terms.append(-arg if hasattr(arg, 'could_extract_minus_sign') else abs(arg))
                else:
                    pos_terms.append(arg)
            
            if len(neg_terms) == 1 and len(pos_terms) == 1:
                pos_tree = _expr_to_tree(pos_terms[0])
                neg_tree = _expr_to_tree(neg_terms[0])
                node_content = f"[$-$\n    {pos_tree}\n    {neg_tree}\n]"
                return _add_color_attrs(node_content, has_children=True)
            else:
                # Multiple terms - construct addition tree
                all_terms = pos_terms + [-term for term in neg_terms]
                if len(all_terms) > 2:
                    # For multiple terms, create nested additions
                    result_tree = _expr_to_tree(all_terms[0])
                    for term in all_terms[1:]:
                        term_tree = _expr_to_tree(term)
                        node_content = f"[$+$\n    {result_tree}\n    {term_tree}\n]"
                        result_tree = _add_color_attrs(node_content, has_children=True)
                    return result_tree
                else:
                    left_tree = _expr_to_tree(all_terms[0])
                    right_tree = _expr_to_tree(all_terms[1])
                    node_content = f"[$+$\n    {left_tree}\n    {right_tree}\n]"
                    return _add_color_attrs(node_content, has_children=True)
        
        # Handle unary minus
        if expr.func == sympy.Mul and len(expr.args) >= 2 and expr.args[0] == -1:
            inner_expr = sympy.Mul(*expr.args[1:])
            inner_tree = _expr_to_tree(inner_expr)
            node_content = f"[$-$\n    {inner_tree}\n]"
            return _add_color_attrs(node_content, has_children=True)
        
        # Handle multiplication with multiple arguments
        if expr.func == sympy.Mul and len(expr.args) > 2:
            # Create nested multiplications
            result_tree = _expr_to_tree(expr.args[0])
            for arg in expr.args[1:]:
                arg_tree = _expr_to_tree(arg)
                node_content = f"[$\\times$\n    {result_tree}\n    {arg_tree}\n]"
                result_tree = _add_color_attrs(node_content, has_children=True)
            return result_tree
        
        # Handle unary operations
        if len(expr.args) == 1:
            arg_tree = _expr_to_tree(expr.args[0])
            node_content = f"[${str(expr.func)}$\n    {arg_tree}\n]"
            return _add_color_attrs(node_content, has_children=True)
        
        # Fallback for unknown expressions
        expr_str = str(expr)
        return _add_color_attrs(f"[${expr_str}$]", has_children=False)
    
    return _expr_to_tree(expr)


def equations_to_latex_forest(equations: Union[str, List[sympy.Expr]], 
                            colors: List[str] = None, 
                            labels: List[str] = None,
                            include_preamble: bool = True,
                            separate_environments: bool = True) -> str:
    """
    Convert equation(s) to LaTeX forest format.
    
    Parameters:
    -----------
    equations : Union[str, List[sympy.Expr]]
        Either equation string from CSV or list of sympy expressions
    colors : List[str], optional
        Colors for highlighting each equation tree
    labels : List[str], optional
        Labels for each tree (e.g., ['drift', 'diffusion'])
    include_preamble : bool
        Whether to include the forest environment preamble
    separate_environments : bool
        Whether to create separate forest environments for each tree (recommended for multiple trees)
        
    Returns:
    --------
    str
        LaTeX forest representation
    """
    
    # Parse equations if needed
    if isinstance(equations, str):
        sympy_exprs = parse_equation_string(equations)
    else:
        sympy_exprs = equations
    
    # Set default colors and labels
    if colors is None:
        colors = ['black', 'red', 'blue', 'ForestGreen', 'orange']
    if labels is None:
        labels = ['drift', 'diffusion'] if len(sympy_exprs) == 2 else [f'eq_{i+1}' for i in range(len(sympy_exprs))]
    
    # Forest environment template
    forest_preamble = """\\begin{forest}
            for tree={
            edge path={
                \\noexpand\\path [\\forestoption{edge}]
                (!u.parent anchor) -- +(0,-5pt) -| (.child anchor)\\forestoption{edge label};
            },
            edge={line width=1pt},
            parent anchor=south,
            child anchor=north
            }
        """
    
    latex_content = ""
    
    # Handle multiple trees with separate environments or single environment
    if len(sympy_exprs) > 1 and separate_environments:
        # Create separate forest environment for each tree
        for i, expr in enumerate(sympy_exprs):
            color = colors[i % len(colors)] if colors[i % len(colors)] != 'black' else None
            label = labels[i] if i < len(labels) else f'eq_{i+1}'
            
            tree = sympy_to_latex_tree(expr, color)
                        
            # Add label
            if len(labels) > i:
                bracket_pos = tree.find('[')
                if bracket_pos != -1:
                    label_attr = f", label={{[xshift=-20pt,yshift=-12pt]}}"
                    first_dollar = tree.find('$', bracket_pos)
                    if first_dollar != -1:
                        second_dollar = tree.find('$', first_dollar + 1)
                        if second_dollar != -1:
                            insert_pos = second_dollar + 1
                            tree = tree[:insert_pos] + label_attr + tree[insert_pos:]
            
            if include_preamble:
                latex_content += forest_preamble
            
            latex_content += tree + "\n"
            
            if include_preamble:
                latex_content += "\\end{forest}\n\\quad\n"
    
    else:
        # Single forest environment for all trees (original behavior)
        if include_preamble:
            latex_content += forest_preamble
        
        for i, expr in enumerate(sympy_exprs):
            color = colors[i % len(colors)] if colors[i % len(colors)] != 'black' else None
            label = labels[i] if i < len(labels) else f'eq_{i+1}'
            
            tree = sympy_to_latex_tree(expr, color)
            
            # Add label if there are multiple expressions
            if len(sympy_exprs) > 1 and len(labels) > i:
                bracket_pos = tree.find('[')
                if bracket_pos != -1:
                    label_attr = f", label={{[xshift=-20pt,yshift=-12pt]}}"
                    first_dollar = tree.find('$', bracket_pos)
                    if first_dollar != -1:
                        second_dollar = tree.find('$', first_dollar + 1)
                        if second_dollar != -1:
                            insert_pos = second_dollar + 1
                            tree = tree[:insert_pos] + label_attr + tree[insert_pos:]
            
            latex_content += tree
            
            if i < len(sympy_exprs) - 1:
                latex_content += "\n\\color{gray}\\vrule\\color{black}\n"
            else:
                latex_content += "\n"
        
        if include_preamble:
            latex_content += "\\end{forest}"
    
    return latex_content


def visualize_equation_evolution(df: pd.DataFrame, 
                                generation_list: List[int] = None, 
                                colors: List[str] = None) -> str:
    """
    Create LaTeX document showing evolution of equations over generations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'generation' and 'equation' columns
    generation_list : List[int], optional
        Specific generations to show. If None, shows all generations
    colors : List[str], optional
        Colors to use for highlighting different expressions
        
    Returns:
    --------
    str
        Complete LaTeX document
    """
    if generation_list is None:
        generation_list = sorted(df['generation'].unique())
    
    if colors is None:
        colors = ['black', 'red', 'blue', 'ForestGreen', 'orange', 'purple']
    
    latex_doc = """\\documentclass{article}
            \\usepackage{forest}
            \\usepackage{xcolor}
            \\usepackage{amsmath}

            \\begin{document}

            \\title{Evolution of Equations Over Generations}
            \\maketitle

            """
    
    for i, gen in enumerate(generation_list):
        gen_data = df[df['generation'] == gen]
        if len(gen_data) == 0:
            continue
            
        equation = gen_data.iloc[0]['equation']
        
        latex_doc += f"\\section*{{Generation {gen}}}\n\n"
        
        # Generate tree for this generation
        tree_latex = equations_to_latex_forest(
            equation, 
            colors=[colors[i % len(colors)], colors[(i+1) % len(colors)]],
            include_preamble=True,
            separate_environments=True  # Use separate environments for cleaner output
        )
        
        latex_doc += tree_latex + "\n\n"
        
        if i < len(generation_list) - 1:
            latex_doc += "\\vspace{1cm}\n\n"
    
    latex_doc += "\\end{document}"
    
    return latex_doc


def save_latex_to_file(latex_content: str, filename: str, path: str = "../figures/"):
    """
    Save LaTeX content to a file.
    
    Parameters:
    -----------
    latex_content : str
        The LaTeX content to save
    filename : str
        Name of the file (without extension)
    path : str
        Directory to save the file
    """
    os.makedirs(path, exist_ok=True)
    
    if not filename.endswith('.tex'):
        filename += '.tex'
    
    filepath = os.path.join(path, filename)
    
    with open(filepath, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX saved to: {filepath}")


def create_multiple_tree_document(equations_list: List[str],
                                 colors_list: List[List[str]] = None,
                                 labels_list: List[List[str]] = None,
                                 titles: List[str] = None) -> str:
    """
    Create a LaTeX document with multiple separate tree groups.
    
    Parameters:
    -----------
    equations_list : List[str]
        List of equation strings, each can contain multiple expressions
    colors_list : List[List[str]], optional
        List of color lists for each equation group
    labels_list : List[List[str]], optional
        List of label lists for each equation group  
    titles : List[str], optional
        Titles for each equation group
        
    Returns:
    --------
    str
        Complete LaTeX document with multiple tree groups
    """
    if colors_list is None:
        colors_list = [['black', 'red', 'blue'] for _ in equations_list]
    if labels_list is None:
        labels_list = [['drift', 'diffusion'] if '[' in eq and ',' in eq else ['equation'] 
                      for eq in equations_list]
    if titles is None:
        titles = [f"Equation Group {i+1}" for i in range(len(equations_list))]
    
    latex_doc = """\\documentclass{article}
\\usepackage{forest}
\\usepackage{xcolor}
\\usepackage{amsmath}

\\begin{document}

"""
    
    for i, (equation, colors, labels, title) in enumerate(
        zip(equations_list, colors_list, labels_list, titles)
    ):
        latex_doc += f"\\section*{{{title}}}\n\n"
        
        # Generate trees with separate environments
        tree_latex = equations_to_latex_forest(
            equation,
            colors=colors,
            labels=labels,
            include_preamble=True,
            separate_environments=True
        )
        
        latex_doc += tree_latex + "\n"
        
        if i < len(equations_list) - 1:
            latex_doc += "\\vspace{1cm}\n\n"
    
    latex_doc += "\\end{document}"
    return latex_doc


def integrate_with_gp_strategy(strategy, candidate, colors=['black', 'red'], 
                             separate_environments: bool = True):
    """
    Integrate with existing GP strategy to convert trees directly to LaTeX.
    
    Parameters:
    -----------
    strategy : GeneticProgramming instance
        GP strategy object that has expression_to_string method
    candidate : Array
        Candidate solution from GP system
    colors : List[str]
        Colors for visualization
        
    Returns:
    --------
    str
        LaTeX forest representation
    """
    # Use the existing expression_to_string method from GP system
    expression_strings = strategy.expression_to_string(candidate)
    
    # Convert to our format
    if isinstance(expression_strings, list):
        equation_string = '[' + ', '.join([str(expr) for expr in expression_strings]) + ']'
    else:
        equation_string = str(expression_strings)
    
    return equations_to_latex_forest(equation_string, colors=colors, 
                                    separate_environments=separate_environments)


def analyze_equation_complexity(equation_str: str) -> dict:
    """
    Analyze the complexity of an equation.
    
    Parameters:
    -----------
    equation_str : str
        The equation string to analyze
        
    Returns:
    --------
    dict
        Dictionary containing complexity metrics
    """
    # Parse equation
    eq_str = equation_str.strip()
    if eq_str.startswith('[') and eq_str.endswith(']'):
        eq_str = eq_str[1:-1]
    
    parts = [part.strip() for part in eq_str.split(',')]
    
    complexity_info = {
        'num_expressions': len(parts),
        'expressions': [],
        'total_nodes': 0,
        'operations': [],
        'variables': set(),
        'constants': []
    }
    
    for part in parts:
        try:
            part_processed = part.replace('laplacian', 'L')
            expr = sympy.sympify(part_processed)
            
            # Count nodes (rough estimate)
            expr_str = str(expr)
            nodes = len([op for op in ['+', '-', '*', '/', '**'] if op in expr_str]) + 1
            complexity_info['total_nodes'] += nodes
            
            # Find operations
            operations = [op for op in ['+', '-', '*', '/', '**'] if op in expr_str]
            complexity_info['operations'].extend(operations)
            
            # Find variables
            for symbol in expr.free_symbols:
                complexity_info['variables'].add(str(symbol))
            
            # Find constants (numbers)
            for atom in expr.atoms():
                if atom.is_number:
                    complexity_info['constants'].append(float(atom))
            
            complexity_info['expressions'].append({
                'expression': str(expr),
                'nodes': nodes,
                'variables': [str(s) for s in expr.free_symbols],
                'constants': [float(a) for a in expr.atoms() if a.is_number]
            })
            
        except Exception as e:
            print(f"Could not analyze expression '{part}': {e}")
    
    complexity_info['variables'] = list(complexity_info['variables'])
    complexity_info['unique_operations'] = list(set(complexity_info['operations']))
    
    return complexity_info

if __name__ == "__main__":
    # Example usage
    example_equation = "[u_xx + u_x, 0.101*u]"
    
    # Generate with separate environments (new default for multiple trees)
    result = equations_to_latex_forest(example_equation, 
                                                   colors=['blue', 'red'],
                                                   show_preview=True,
                                                   separate_environments=True)
    
    print(result['latex'])