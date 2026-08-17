"""
1. Code Flow & Logic:
  * Hierarchical Comparison: The script uses a recursive function find_diff_paths to traverse the YAML structure of multiple experiments simultaneously. It identifies "leaf" nodes where values differ
    across experiments.
  * Discovery: It walks the project directory to find experiment folders matching a timestamp pattern (YYYY-MM-DD_HH-MM-SS).
  * Extraction: Currently, it loads env.yaml and extracts top-level keys. It handles the observations key specially by creating separate tabs for its children (e.g., obs_policy, obs_critic).
  * Reporting: For each identified YAML node (task), it creates an Excel sheet. Columns are formed by the paths to differing parameters, and rows represent individual experiments.


 2. Methods Used:
  * find_diff_paths: The core comparison engine. It skips the func key and focuses on identifying paths with non-identical values.
  * get_value_at_path: A helper to retrieve nested values given a tuple path.
  * pd.ExcelWriter: Used to aggregate multiple DataFrames into a single multi-tab spreadsheet.
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import yaml


class GenericLoader(yaml.SafeLoader):
    pass


def construct_undefined(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return None


GenericLoader.add_constructor(None, construct_undefined)


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


GenericLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_python_tuple)


def find_diff_paths(values, current_path=()):
    """
    Recursively traverses a list of objects (from different experiments)
    simultaneously to find paths to leaf nodes that differ.
    """
    # 1. Check if all values are identical
    if len(set(map(str, values))) <= 1:
        return []

    # 2. Check if we can/should go deeper
    any_dict = any(isinstance(v, dict) for v in values)

    if any_dict:
        # Collect all unique keys at this level across all experiments
        keys = set()
        for v in values:
            if isinstance(v, dict):
                keys.update(v.keys())

        all_paths = []
        for k in sorted(keys):
            if k == "func":
                continue
            # Collect values for this specific key from all experiments
            sub_values = [v.get(k) if isinstance(v, dict) else None for v in values]
            all_paths.extend(find_diff_paths(sub_values, current_path + (k,)))

        # If values were different but no sub-paths found
        if not all_paths and len(set(map(str, values))) > 1:
            return [current_path]
        return all_paths
    else:
        # Different values at a leaf
        return [current_path]


def get_value_at_path(data, path):
    """Retrieves a value from a nested dict using a path of keys."""
    for key in path:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return "N/A"
    return data


def is_experiment_dir(name):
    return re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name) is not None


def main():
    parser = argparse.ArgumentParser(
        description="Efficiently compare RL experiments using hierarchical tree traversal."
    )
    parser.add_argument("root_dir", help="Root directory to search for experiments")
    parser.add_argument(
        "--output", default="experiment_comparison.xlsx", help="Output Excel file name"
    )
    args = parser.parse_args()

    experiments = []
    all_top_keys = set()
    all_obs_sub_keys = set()

    print(f"Searching for experiments in {args.root_dir}...")

    for root, dirs, files in os.walk(args.root_dir):
        for d in dirs:
            if is_experiment_dir(d):
                exp_path = os.path.join(root, d)
                env_yaml_path = os.path.join(exp_path, "params", "env.yaml")
                agent_yaml_path = os.path.join(exp_path, "params", "agent.yaml")

                exp_entry = {"id": os.path.relpath(exp_path, args.root_dir)}

                # Load env.yaml
                if os.path.exists(env_yaml_path):
                    try:
                        with open(env_yaml_path, "r") as f:
                            env_data = yaml.load(f, Loader=GenericLoader)
                        exp_entry["env_data"] = env_data
                        if isinstance(env_data, dict):
                            all_top_keys.update(env_data.keys())
                            obs_data = env_data.get("observations")
                            if isinstance(obs_data, dict):
                                all_obs_sub_keys.update(obs_data.keys())
                    except Exception as e:
                        print(f"Error loading {env_yaml_path}: {e}")

                # Load agent.yaml
                if os.path.exists(agent_yaml_path):
                    try:
                        with open(agent_yaml_path, "r") as f:
                            agent_data = yaml.load(f, Loader=GenericLoader)
                        exp_entry["agent_data"] = agent_data
                    except Exception as e:
                        print(f"Error loading {agent_yaml_path}: {e}")

                if "env_data" in exp_entry or "agent_data" in exp_entry:
                    experiments.append(exp_entry)

    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiments.")

    # Categorize agent.yaml data
    agent_special_keys = ["policy", "critic", "encoder"]
    for exp in experiments:
        agent_data = exp.get("agent_data", {})
        if isinstance(agent_data, dict):
            exp["experiment_details"] = {
                k: v for k, v in agent_data.items() if k not in agent_special_keys
            }
        else:
            exp["experiment_details"] = {}

    # Define tasks: (Sheet Name, Base Hierarchy Path, Source Key)
    tasks = []

    # Env tasks
    for key in sorted(all_top_keys):
        if key == "observations":
            for sub_key in sorted(all_obs_sub_keys):
                tasks.append((f"obs_{sub_key}", ("observations", sub_key), "env_data"))
        else:
            tasks.append((key, (key,), "env_data"))

    # Agent tasks
    tasks.append(("policy", ("policy",), "agent_data"))
    tasks.append(("critic", ("critic",), "agent_data"))
    tasks.append(("encoder", ("encoder",), "agent_data"))
    tasks.append(("experiment details", (), "experiment_details"))

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for sheet_name, base_path, source_key in tasks:
            # Extract the relevant sub-tree from each experiment
            sub_trees = [
                get_value_at_path(exp.get(source_key, {}), base_path)
                for exp in experiments
            ]

            # Find all relative paths within this sub-tree that differ across experiments
            diff_relative_paths = find_diff_paths(sub_trees)

            if not diff_relative_paths:
                continue

            # Build data for this sheet
            data_list = []
            for i, exp in enumerate(experiments):
                row = {"experiment_id": exp["id"]}
                tree = sub_trees[i]
                for rel_path in diff_relative_paths:
                    col_name = ".".join(rel_path) if rel_path else "value"
                    row[col_name] = get_value_at_path(tree, rel_path)
                data_list.append(row)

            df = pd.DataFrame(data_list)
            cols = ["experiment_id"] + [c for c in df.columns if c != "experiment_id"]
            df = df[cols]

            safe_sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            print(
                f"Created sheet: {safe_sheet_name} with {len(df.columns) - 1} differing parameters."
            )

    print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
