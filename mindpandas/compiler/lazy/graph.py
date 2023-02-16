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
"""
This module defines graph used in MinSpore pandas lazy mode.
"""
import networkx as nx

from .query_plan import Function, Operator


class DirectedGraph:
    """
    This class holds the nodes of all the pandas objects in the session.
    """

    def __init__(self):
        # self.graph = nx.MultiDiGraph()
        self.graph = nx.DiGraph()
        self._node_id = 0
        self._edge_id = 0

    def _incr_node_id(self):
        # Need a lock to make the increment thread safe
        self._node_id += 1

    def _incr_edge_id(self):
        # Need a lock to make the increment thread safe
        self._edge_id += 1

    def add_node(self, name, fn, stats, **kwargs):
        self._incr_node_id()
        self.graph.add_node(self._node_id, id=self._node_id,
                            name=name, func=fn, stats=stats, args=kwargs)
        return self._node_id

    def clone_node(self, src_node):
        self._incr_node_id()
        self.graph.add_node(self._node_id, id=self._node_id, name=src_node['name'], func=src_node['func'],
                            stats=src_node['stats'], args=src_node['args'])
        return self._node_id

    def add_edge(self, src, dest):
        self._incr_edge_id()
        self.graph.add_edge(src, dest, id=self._edge_id,
                            args={'from': src, 'to': dest})

    def remove_edge(self, src, dest):
        self.graph.remove_edge(src, dest)

    def add_source_node(self, name, fn, stats=None, **kwargs):
        return self.add_node(name, fn, stats, **kwargs)

    def add_1op_node(self, name, fn, child, stats=None, **kwargs):
        new_node_id = self.add_node(name, fn, stats, **kwargs)
        self.add_edge(new_node_id, child)
        return new_node_id

    def add_2op_node(self, name, fn, left_child, right_child, stats=None, **kwargs):
        new_node_id = self.add_node(name, fn, stats, **kwargs)
        self.add_edge(new_node_id, left_child)
        self.add_edge(new_node_id, right_child)
        return new_node_id

    def add_op_node(self, name, fn, children: list, stats=None, **kwargs):
        new_node_id = self.add_node(name, fn, stats, **kwargs)
        for node_id in children:
            self.add_edge(new_node_id, node_id)
        return new_node_id

    def edges(self, node_id):
        return self.graph.edges(node_id)

    def number_of_nodes(self):
        return self.graph.number_of_nodes

    def postorder_nodes(self, node_id):
        return list(nx.dfs_postorder_nodes(self.graph, node_id))

    def preorder_nodes(self, node_id):
        return list(nx.dfs_preorder_nodes(self.graph, node_id))

    def bfs_nodes(self, node_id):
        return dict(nx.bfs_successors(self.graph, node_id)).keys()

    def children(self, node_id):
        try:
            return nx.dfs_successors(self.graph, source=node_id, depth_limit=1)[node_id]
        except (KeyError, ValueError, TypeError):
            return []

    def orphan(self, node_id):
        return self.graph.in_degree(node_id) == 0

    def cse(self, node_id):
        return self.graph.in_degree(node_id) > 1

    def parents(self, node_id):
        res = []
        for p in self.graph.in_edges(node_id):
            res.append(p[0])
        return res

    def print(self, node_id, pr_stats=False, pr_details=False, level=0):
        return self._print_op(node_id, pr_stats, pr_details, level)

    def _print_op(self, node_id, pr_stats, pr_details, level):
        """
        Walk the plan top-down from the input node and for each level,
        print the operator and its operand(s).
        """
        prefix = '+- '
        cse_prefix = '*- '
        indent = '|  '

        out: str = ""
        for _ in range(level):
            out += indent

        if self.cse(node_id):
            out += cse_prefix
        else:
            out += prefix
        pairs = self.graph.nodes[node_id]
        out += "[{0}] {1}({2})".format(pairs['id'],
                                       pairs['name'], pairs['func'])
        if pr_stats:
            try:
                stats = pairs['stats']
            except KeyError:
                stats = None

            if stats is not None:
                out += " {0}".format(stats)
        if pr_details and pairs['args']:
            out += " args={0}".format(pairs['args'])
        if 'executor' in pairs:
            out += " executor={0}".format(pairs['executor'])
        out += '\n'
        if pairs['name'] in [Operator.TASKCHAIN, Operator.REGION]:
            plan = pairs['func']
            # out += "before print task/region"
            out += plan.print(plan.root, pr_details=pr_details,
                              pr_stats=pr_stats, level=level+2)
            for _ in range(level):
                out += indent
            if self.cse(node_id):
                out += cse_prefix
            else:
                out += prefix
            out += "[{0}] END {1}({2})".format(pairs['id'],
                                               pairs['name'], pairs['func'])
            out += '\n'
        if pairs['func'] in [Function.TASKCHAIN]:
            plan = pairs['args']['_graph']
            # out += "before print task/region"
            out += plan.print(plan.root, pr_details=pr_details,
                              pr_stats=pr_stats, level=level+2)
            for _ in range(level):
                out += indent
            if self.cse(node_id):
                out += cse_prefix
            else:
                out += prefix
            out += "[{0}] END {1}({2})".format(pairs['id'],
                                               pairs['name'], pairs['func'])
            out += '\n'
        descendants = self.graph.successors(node_id)
        for child in descendants:
            level += 1
            # Add indentation
            # for i in range(level):
            #    out += indent
            out += self._print_op(child, pr_stats, pr_details, level)
            level -= 1

        return out

    def set_stats(self, node_id, stats):
        self.graph.nodes[node_id]['stats'] = stats

    def stats(self, node_id):
        try:
            stats = self.graph.nodes[node_id]['stats']
        except KeyError:
            raise Warning(
                "Statistics of node {0} has not yet populated.".format(node_id))
        else:
            return stats

    def cache(self, node_id, result):
        self.graph.nodes[node_id]['result'] = result

    def result(self, node_id):
        try:
            result = self.graph.nodes[node_id]['result']
        except KeyError:
            return None
        else:
            return result

    def copy_graph(self, plan, nodes=None):
        if nodes is None:
            nodes = plan.preorder_nodes(plan.root)

        self.graph = plan.graph.subgraph(nodes).copy()
        self.root = nodes[0]

    def create_copy(self):
        """
        Use to create deep copy of plan/graph object
        Recommended for non-dist duplication of subgraphs
        """
        new_graph = type(self)()
        nodes = self.preorder_nodes(self.root)
        new_graph.graph = self.graph.subgraph(nodes).copy()
        new_graph.root = self.root
        return new_graph

    def replace_node(self, node_id, new_node_id):
        for parent in self.parents(node_id):
            self.add_edge(parent, new_node_id)
            self.remove_edge(parent, node_id)
        for child in self.children(node_id):
            self.add_edge(new_node_id, child)
            self.remove_edge(node_id, child)
        if node_id == self.root:
            self.root = new_node_id

    def delete_node(self, node_id, bypass=False):
        """
        delete node from graph.
        """
        parents = self.parents(node_id)
        children = self.children(node_id)
        if not bypass:
            assert (len(parents) == len(children) or not
                    children or not parents)
            if parents:
                for i in range(len(children)):
                    self.add_edge(parents[i], children[i])
        else:
            assert (len(parents) <= 1 or len(children) <= 1)
            if len(parents) == 1:
                for i in range(len(children)):
                    self.add_edge(parents[0], children[i])
            if len(children) == 1:
                for i in range(len(parents)):
                    self.add_edge(parents[i], children[0])
        # assert(len(parents) <= len(children) or len(children) == 0)

        for parent in parents:
            self.remove_edge(parent, node_id)
        for child in self.children(node_id):
            self.remove_edge(node_id, child)
        if node_id == self.root:
            if len(children) == 1:
                self.root = children[0]
            else:
                print(
                    "deleting root note that has more than one child.  Do not know which to make new root!!")

    def move_node_to_plan_at_root(self, node_id, inner_plan):
        """
        check if inner_plan root is AUX
        in this case we have to add the node in the same level as AUX
        """
        root = inner_plan.graph.nodes[inner_plan.root]
        is_root_aux = root['name'] == Operator.AUX

        if is_root_aux and inner_plan.root_2 is None:
            node = self.graph.nodes[node_id]
            inner_plan.graph.add_node(node_id, id=node_id, name=node['name'], func=node['func'],
                                      stats=node['stats'], args=node['args'])

            # find root's children
            children = inner_plan.children(inner_plan.root)
            assert len(children) == 1
            inner_plan.add_edge(node_id, children[0])

            inner_plan.root_2 = inner_plan.root
            inner_plan.root = node_id

            # add root nodes:
            self.delete_node(node_id, bypass=True)

        else:
            node = self.graph.nodes[node_id]
            # inner_plan.add_edge(node_id, inner_plan.root)
            inner_plan.graph.add_node(node_id, id=node_id, name=node['name'], func=node['func'],
                                      stats=node['stats'], args=node['args'])

            # add node_id's children to the inner plan:
            children = self.children(node_id)
            for child_id in children:
                child_node = self.graph.nodes[child_id]
                child_fn = child_node['func']
                if child_fn == Function.AUX:
                    inner_plan.graph.add_node(child_id, id=child_id, name=child_node['name'], func=child_node['func'],
                                              stats=child_node['stats'], args=child_node['args'])
                    inner_plan.add_edge(node_id, child_id)
                    self.remove_edge(node_id, child_id)

                    # dfs_nodes = inner_plan.postorder_nodes(inner_plan.root)
                    # inner_plan.add_edge(dfs_nodes[0], child_id)
                    break

            if inner_plan.root_2:
                inner_plan.add_edge(node_id, inner_plan.root_2)
            inner_plan.add_edge(node_id, inner_plan.root)

            inner_plan.root = node_id
            inner_plan.root_2 = None
            self.delete_node(node_id, bypass=True)

    def longest_paths(self):
        graph_copied = self.graph.copy()
        paths = []
        while graph_copied.nodes:
            longest = nx.dag_longest_path(graph_copied)
            paths.append(longest)
            graph_copied.remove_nodes_from(longest)
        return paths


class ValidatedPlan(DirectedGraph):
    """
    This class is a view of a subgraph of the nodes reachable from the input root node
    """

    def __init__(self, dag, root_id):
        # Do NOT call super().__init__
        # We want to reuse a subgraph from its parent
        graph = dag.graph
        self.root = root_id
        nodes = dag.preorder_nodes(root_id)
        self.graph = graph.subgraph(nodes)


class LogicalPlan(DirectedGraph):
    """
    This class represents a logical plan in the rewrite phase
    """

    def __init__(self):
        super().__init__()
        self.root = None
        self.root_2 = None


class PhysicalPlan(DirectedGraph):
    """
    This class represents a physical plan in the optimize phase
    """

    def __init__(self):
        super().__init__()
        self.root = None
        self.root_list = []  # list of roots
        # a dictionary of "nodes that their results should be communicated" to their desttination
        self.dest_map = {}


class ExecutionPlan(DirectedGraph):
    """
    This class represents a physical plan in the optimize phase
    """

    def __init__(self):
        super().__init__()
        self.root = None
        # a dictionary of "nodes that their results should be communicated" to their desttination
        self.dest_map = {}

    def add_executor(self, node_id, executor_handle, executor_id):
        self.graph.nodes[node_id]['executor'] = (executor_handle, executor_id)

    def get_executor(self, node_id):
        try:
            executor_handle, executor_id = self.graph.nodes[node_id]['executor']
        except KeyError:
            return None
        else:
            return executor_handle, executor_id
