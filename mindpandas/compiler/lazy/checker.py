# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindPandas Lazy Mode Checker Class"""

from .graph import ValidatedPlan
from .query_plan import Function
from .query_plan import Operator


class Checker:
    """
    This class performs semantic checking on a DAG-based logical plan.

    Args:
        workspace (Workspace): The workspace containing the DAG.
        root (str): The root node of the DAG.
        debug (bool): Enable or disable debugging messages.
        pr_stats (bool): Print statistics.
        pr_details (bool): Print details.
    """

    def __init__(self, workspace, root, debug, pr_stats, pr_details):
        self.ws = workspace
        plan = ValidatedPlan(workspace.dag, root)
        self.plan = plan
        self.graph = plan.graph
        self.root = plan.root
        self.debug = debug
        self.pr_stats = pr_stats
        self.pr_details = pr_details
        if self.debug:
            print("=== Logical Plan (entering Checker) ===")
            print(self.plan.print(self.root, self.pr_stats, self.pr_details))

    def check_read(self, node_id):
        """Checks if the specified node is a read operation and performs semantic checks."""

    def check_map(self, node_id):
        """Checks if the specified node is a map operation and performs semantic checks."""

    def check_groupby(self, node_id):
        """Checks if the specified node is a groupby operation and performs semantic checks."""
        node = self.graph.nodes[node_id]
        assert node['name'] == Operator.GROUPBY
        child_node_id = self.plan.children(node_id)[0]
        child = self.graph.nodes[child_node_id]
        if child['func'] == Function.DATAFRAME:
            axis = node['args']['axis']
            groupby_columns = node['args']['by']
            if groupby_columns is not None:
                if axis == 0:
                    columns = child['args']['columns']
                    groupby_columns = [groupby_columns] if not isinstance(groupby_columns, list) else groupby_columns
                    if any(col not in columns for col in groupby_columns):
                        msg = str([col for col in groupby_columns if col not in columns]).replace(",", "")
                        raise RuntimeError(f"Groupby columns {msg} not in the input DataFrame")

    def check_agg(self, node_id):
        """Checks if the specified node is an aggregate function operation and performs semantic checks."""

    def check_join(self, node_id):
        """Checks if the specified node is a join operation and performs semantic checks."""

    def check_filter(self, node_id):
        """Checks if the specified node is a filter operation and performs semantic checks."""

    def run(self):
        """Runs semantic checks on all nodes in the DAG and returns the validated plan."""
        # A collection of semantics checking rules
        rule_map = {
            Operator.SOURCE: self.check_read,
            Operator.SCALARF: self.check_map,
            Operator.GROUPBY: self.check_groupby,
            Operator.AGGF: self.check_agg,
            Operator.JOIN: self.check_join,
            Operator.FILTER: self.check_filter,
        }
        node_ids = self.plan.postorder_nodes(self.root)
        for node_id in node_ids:
            name = self.graph.nodes[node_id]['name']
            try:
                checker = rule_map[name]
            except KeyError:
                continue
            else:
                # Run the checker on the current node
                if self.ws.is_disable("chk", name, checker.__name__):
                    continue
                checker(node_id)

        # If no semantics error found, return the validated plan
        return self.plan
