"""XML Route File Splitter for Parallel CARLA Evaluation.

This utility splits a single CARLA routes XML file into multiple smaller files
for parallel evaluation. Useful for distributing evaluation workload across
multiple processes or machines.

Usage:
    python split_xml.py <base_route> <task_num> <algo> <planner_type>

Example:
    python split_xml.py leaderboard/data/routes 4 simlingo default
    This creates 4 files: routes_0_simlingo_default.xml, routes_1_simlingo_default.xml, etc.
"""

import xml.etree.ElementTree as ET


def split_list_into_n_parts(lst, n):
    """Split a list into n roughly equal parts.

    Uses divmod to ensure parts are as evenly sized as possible.

    Args:
        lst: List to split.
        n: Number of parts to create.

    Returns:
        Generator yielding n sublists.
    """
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main(base_route, task_num, algo, planner_type):
    """Split an XML routes file into multiple files for parallel processing.

    Args:
        base_route: Base path/name of the route XML file (without .xml extension).
        task_num: Number of splits to create.
        algo: Algorithm/model name to include in output filenames.
        planner_type: Planner type to include in output filenames.
    """
    # Parse the original XML routes file
    tree = ET.parse(f'{base_route}.xml')
    root = tree.getroot()

    # Extract all route elements
    case = root.findall('route')

    # Split routes into n parts
    results = split_list_into_n_parts(case, task_num)

    # Create a new XML file for each split
    for index, re in enumerate(results):
        # Create new root element
        new_root = ET.Element("routes")

        # Add routes to new root
        for x in re:
            new_root.append(x)

        # Write to file with descriptive name
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f'{base_route}_{index}_{algo}_{planner_type}.xml',
                      encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_route", type=str, help="Base route filename (without .xml)")
    parser.add_argument("task_num", type=int, help="Number of files to split into")
    parser.add_argument("algo", type=str, help="Algorithm/model name")
    parser.add_argument("planner_type", type=str, help="Planner type identifier")
    args = parser.parse_args()
    main(args.base_route, args.task_num, args.algo, args.planner_type)