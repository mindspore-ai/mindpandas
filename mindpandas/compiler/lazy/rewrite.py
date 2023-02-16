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
"""MindPandas Lazy Mode Rewrite Class"""
import math

from .graph import LogicalPlan
from .statistics import Statistics
from .query_plan import Function, Operator, fn_op_map


class Rewrite:
    """Rewrite class for rewrite the logical plan"""

    def __init__(self, workspace, plan, debug, pr_stats, pr_details):
        self.ws = workspace
        self.debug = debug
        self.pr_stats = pr_stats
        self.pr_details = pr_details
        # Create a new logical plan
        self.rwr_plan = rwr_plan = LogicalPlan()

        in_graph = plan.graph
        in_root = plan.root
        dfs_nodes = plan.preorder_nodes(plan.root)
        old_new_map = dict()
        # walk DFS
        for node_id in dfs_nodes:
            node = in_graph.nodes[node_id]
            new_node_id = rwr_plan.clone_node(node)
            # Add statistics object to each new node
            self.add_statistics_object(new_node_id)
            res = plan.result(node_id)
            if res is not None:
                rwr_plan.cache(new_node_id, res)

            old_new_map[node_id] = new_node_id
            if node_id == in_root:
                rwr_plan.root = new_node_id

        for node_id in dfs_nodes:
            edges = plan.edges(node_id)
            for edge in edges:
                new_src_node = old_new_map[edge[0]]
                new_dest_node = old_new_map[edge[1]]
                rwr_plan.add_edge(new_src_node, new_dest_node)

        self.rwr_graph = rwr_plan.graph
        self.rwr_root = rwr_plan.root
        if self.debug:
            print("=== Logical Plan (entering Rewrite) ===")
            print(self.rwr_plan.print(self.rwr_root,
                                      self.pr_stats, self.pr_details))

    def get_node(self, node_id):
        node = self.rwr_graph.nodes[node_id]
        op = node['name']
        func = node['func']
        return node, op, func

    def add_statistics_object(self, node_id):
        if self.rwr_plan.stats(node_id) is None:
            self.rwr_plan.set_stats(node_id, Statistics())

    def infer_groupby_cardinality(self, node_id):
        """infer groupby cardinality"""
        child_ids = self.rwr_plan.children(node_id)
        assert len(child_ids) == 1
        child_id = child_ids[0]
        child_stats = self.rwr_plan.stats(child_id)
        assert child_stats is not None
        stats = Statistics()
        if child_stats.ncolumns[0] is not math.nan:
            stats.infer(ncolumns=child_stats.ncolumns[0])
        self.rwr_plan.set_stats(node_id, stats)
        return 1

    def infer_aggfn_cardinality(self, node_id):
        """infer agg function cardinality"""
        cur_node, _, _ = self.get_node(node_id)
        child_ids = self.rwr_plan.children(node_id)
        assert len(child_ids) == 1
        child_id = child_ids[0]
        _, _, child_func = self.get_node(child_id)
        child_stats = self.rwr_plan.stats(child_id)
        assert child_stats is not None
        stats = Statistics()
        if child_func == Function.GROUPBY:
            if child_stats.ncolumns[0] is not math.nan:
                stats.infer(ncolumns=child_stats.ncolumns[0])
            else:
                # Early out, no stats updated
                return -1
        else:
            axis = cur_node['args']['axis']
            if axis == 0 and child_stats.ncolumns[0] is not math.nan:
                stats.infer(nrows=child_stats.ncolumns[0], ncolumns=1)
            elif axis == 1 and child_stats.nrows[0] is not math.nan:
                stats.infer(nrows=child_stats.nrows[0], ncolumns=1)
            else:
                # Early out, no stats updated
                return -1
        self.rwr_plan.set_stats(node_id, stats)
        return 1

    def leaf_node_stats(self, node_id):
        """get the statistic of the leaf node"""
        stats = self.rwr_plan.stats(node_id)
        if stats.ncolumns[0] is math.nan and stats.nrows[0] is math.nan:
            # If the leaf node has no statistics, do an estimate
            stats = Statistics()
            # Just set some fake numbers
            stats.estimate(nrows=101, ncolumns=8)
            self.rwr_plan.set_stats(node_id, stats)
            return 1
        return -1

    def merge_sel_proj(self, node_id):
        """merge selection projection"""
        def check_sel_pattern(children):
            lhs = op = const = None
            if len(children) != 2:
                matched = 0  # failed
                return matched, lhs, op, const

            first_id = children[0]
            pred_id = children[1]
            _, first_op, _ = self.get_node(first_id)
            pred_node, _, pred_func = self.get_node(pred_id)
            if pred_func != Function.COMPOP:
                matched = 0  # failed
                return matched, lhs, op, const

            operands = self.rwr_plan.children(pred_id)
            lhs_id = operands[0]
            lhs_node, _, lhs_func = self.get_node(lhs_id)
            if lhs_func != Function.PROJECT1:
                matched = 0  # failed
                return matched, lhs, op, const

            common_child_ids = self.rwr_plan.children(lhs_id)
            common_child_id = common_child_ids[0]
            if first_op == Operator.PROJECT:
                # merge the select with project
                proj_child_ids = self.rwr_plan.children(first_id)
                proj_child_id = proj_child_ids[0]
                if common_child_id == proj_child_id:
                    """
                    +- [1] Select(select)
                    |  +- [2] Project(project-n) or Project(project-1)
                    |  |  *- [3] <node>
                    |  +- [4] ScalarFn(compop)
                    |  |  +- [5] Project(project-1)
                    |  |  |  *- [3] <same-node-as-above>
                    """
                    matched = 1  # found pattern 1
                    try:
                        _ = self.rwr_graph.nodes[first_id]['args']['prd']
                    except KeyError:
                        # No predicate in the Project, we are good.
                        pass
                    else:
                        # TODO: Future extension to rewrite even there is a pred in Project
                        matched = 0  # failed
                        return matched, lhs, op, const
                else:
                    matched = 0  # failed
                    return matched, lhs, op, const
            elif first_id == common_child_id:
                """
                +- [1] Select(select)
                |  *- [2] <node>
                |  +- [3] ScalarFn(compop)
                |  |  +- [4] Project(project-1)
                |  |  |  *- [2] <same-node-as-above>
                """
                matched = 2  # found pattern 2
            else:
                matched = 0  # failed
                return matched, lhs, op, const

            lhs = lhs_node['args']['column']
            op = pred_node['args']['op']
            const = pred_node['args']['other']
            return matched, lhs, op, const

        _, cur_op, _ = self.get_node(node_id)
        child_ids = self.rwr_plan.children(node_id)
        child_id = child_ids[0]
        _, child_op, _ = self.get_node(child_id)
        if cur_op == Operator.SELECT:
            if child_op == Operator.SELECT:
                # TODO: Future extension to merge the 2 selects
                return -1
            matched, lhs, op, const = check_sel_pattern(child_ids)
            if matched == 1:
                """
                Input:
                +- [1] Select(select)
                |  +- [2] Project(project-n) or Project(project-1)
                |  |  *- [3] <node>
                |  +- [4] ScalarFn(compop)
                |  |  +- [5] Project(project-1)
                |  |  |  *- [3] <same-node-as-above>
                """
                select_id = node_id
                project_id = child_id
                self.rwr_graph.remove_edge(select_id, project_id)
                parents = self.rwr_plan.parents(select_id)
                for p in parents:
                    self.rwr_graph.remove_edge(p, select_id)
                    self.rwr_graph.add_edge(p, project_id)
                if self.rwr_root == select_id:
                    # Re-position the root to the Project
                    self.rwr_root = self.rwr_plan.root = project_id

                """
                Output (after rewrite):
                +- [2] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                |  +- [3] <node>

                If [1] Select is the root node of the plan, we need to set the [2] Project as the new root.
                """
                # Add the pred to the Project
                pred_expr = [op, lhs, const]  # stored in post-order
                project_node, _, _ = self.get_node(project_id)
                project_node['args']['prd'] = pred_expr
                return 1
            if matched == 2:
                """
                +- [1] Select(select)
                |  *- [2] <node>
                |  +- [3] ScalarFn(compop)
                |  |  +- [4] Project(project-1)
                |  |  |  *- [2] <same-node-as-above>
                """
                select_id = node_id
                common_node_id = child_id
                self.rwr_graph.remove_edge(select_id, common_node_id)
                proj_n_id = self.rwr_plan.add_1op_node(name=fn_op_map[Function.PROJECTN], fn=Function.PROJECTN,
                                                       child=common_node_id)
                parents = self.rwr_plan.parents(select_id)
                for p in parents:
                    self.rwr_graph.remove_edge(p, select_id)
                    self.rwr_graph.add_edge(p, proj_n_id)
                if self.rwr_root == select_id:
                    # Re-position the root to the Project
                    self.rwr_root = self.rwr_plan.root = proj_n_id

                """
                Output (after rewrite):
                +- [5] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                |  +- [3] <node>

                Note [5] is a new node. If [1] Select is the root node of the plan,
                we need to set the [5] Project as the new root.
                """
                # Add the pred to the Project
                pred_expr = [op, lhs, const]  # stored in post-order
                project_node, _, _ = self.get_node(proj_n_id)
                project_node['args']['prd'] = pred_expr
                return 1
            # no rewrite
            return -2
        if cur_op == Operator.PROJECT:
            try:
                _ = self.rwr_graph.nodes[node_id]['args']['prd']
            except KeyError:
                # No predicate in the Project, we are good, continue the rewrite.
                pass
            else:
                # TODO: Future extension to rewrite even there is a pred in Project
                return -3

            if child_op == Operator.PROJECT:
                # TODO: Future extension to merge the 2 projects
                return -4
            if child_op == Operator.SELECT:
                # merge the project with select
                project_id = node_id
                select_id = child_id
                sel_child_ids = self.rwr_plan.children(select_id)
                matched, lhs, op, const = check_sel_pattern(sel_child_ids)
                if matched == 2:
                    """
                    Input:
                    +- [1] Project(project-n) or Project(project-1)
                    |  +- [2] Select(select)
                    |  |  *- [3] <node>
                    |  |  +- [4] ScalarFn(compop)
                    |  |  |  +- [5] Project(project-1)
                    |  |  |  |  *- [3] <same-node-as-above>
                    """
                    src_id = sel_child_ids[0]
                    self.rwr_graph.remove_edge(project_id, select_id)
                    self.rwr_graph.add_edge(project_id, src_id)
                    """
                    Output (after rewrite):
                    +- [1] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                    |  +- [3] <node>
                    """
                    # Add the pred to the Project
                    pred_expr = [op, lhs, const]  # stored in post-order
                    project_node, _, _ = self.get_node(project_id)
                    project_node['args']['prd'] = pred_expr
                    return 2
                # no rewrite
                return -4
            # no rewrite
            return -5
        # no rewrite
        return -6

    def inject_op(self, parent_id, child_id, func_type=Function.PROJECTN):
        """
        Inject op
        """
        # Inject a new operator between the parent and the child
        new_node_id = self.rwr_plan.add_1op_node(
            name=fn_op_map[func_type], fn=func_type, child=child_id)
        children_ids = self.rwr_plan.children(parent_id)
        found = False
        # Maintain the correct order of the children of the parent
        for node_id in children_ids:
            if found or node_id == child_id:
                self.rwr_graph.remove_edge(parent_id, node_id)
                found = True
        assert found

        found = False
        for node_id in children_ids:
            if node_id == child_id:
                self.rwr_graph.add_edge(parent_id, new_node_id)
                found = True
            elif found:
                self.rwr_graph.add_edge(parent_id, node_id)
        return new_node_id

    def inject_projectn_op(self, parent_id, child_id=None, out_columns=None, pred=None):
        """
        Inject projectn operation
        """
        if child_id is None:
            # insert above the only child of the parent
            children = self.rwr_plan.children(parent_id)
            assert len(children) == 1
            child_id = children[0]
        new_projectn_id = self.inject_op(
            parent_id, child_id, Function.PROJECTN)
        projectn_node, _, _ = self.get_node(new_projectn_id)
        if out_columns is not None:
            projectn_node['args']['columns'] = out_columns
        if pred is not None:
            projectn_node['args']['prd'] = pred
        return new_projectn_id

    def push_project_under_join(self, node_id):
        """
        Push project under join.
        """
        proj_node, proj_op, _ = self.get_node(node_id)
        join_node_id = self.rwr_plan.children(node_id)[0]
        join_node, join_op, _ = self.get_node(join_node_id)
        if proj_op == Operator.PROJECT and join_op == Operator.JOIN:
            try:
                proj_prd = proj_node['args']['prd']
            except KeyError:
                return -1

            join_type = join_node['args']['how']
            # try:
            #     join_cols = join_node['args']['on']
            # except KeyError:
            #     join_cols = None
            # if join_cols is not None:
            #     left_join_cols = right_join_cols = join_cols
            # else:
            #     try:
            #         left_join_cols = join_node['args']['left_on']
            #     except KeyError:
            #         left_join_cols = None
            #     try:
            #         right_join_cols = join_node['args']['right_on']
            #     except KeyError:
            #         right_join_cols = None

            grandchildren = self.rwr_plan.children(join_node_id)
            assert len(grandchildren) == 2
            left_node_id = grandchildren[0]
            right_node_id = grandchildren[1]
            left_node, _, _ = self.get_node(left_node_id)
            right_node, _, _ = self.get_node(right_node_id)
            try:
                left_out_cols = left_node['args']['columns']
            except KeyError:
                return -2
            if not isinstance(left_out_cols, list):
                left_out_cols = list(left_out_cols)
            try:
                right_out_cols = right_node['args']['columns']
            except KeyError:
                return -3
            if not isinstance(right_out_cols, list):
                right_out_cols = list(right_out_cols)

            # Collecting the columns in the predicate in PROJECT
            # The predicate is stored in post-order form.
            # We assume it's a comparison op in the form of <compop> <left-opnd> <right-opnd>
            assert len(proj_prd) == 3
            proj_prd_cols = []
            if isinstance(proj_prd[1], str):
                proj_prd_cols.append(proj_prd[1])
            if isinstance(proj_prd[2], str):
                proj_prd_cols.append(proj_prd[2])

            proj_prd_in_left = all(
                item in left_out_cols for item in proj_prd_cols)
            proj_prd_in_right = all(
                item in right_out_cols for item in proj_prd_cols)
            # We assume there is no name collision in the output columns from the left operand
            # and the right operand of the join.
            # We will deal with the 'suffix' when there are name collisions later.

            if join_type in ('inner', 'cross'):
                if proj_prd_in_left and not proj_prd_in_right:
                    # Push down Project to the left operand
                    new_node_id = self.inject_projectn_op(
                        join_node_id, left_node_id, left_out_cols, proj_prd)
                    # Just for debugging
                    _, _, _ = self.get_node(new_node_id)
                    # TODO: Mark as optional predicate instead
                    del proj_node['args']['prd']
                    return 1
                if proj_prd_in_right and not proj_prd_in_left:
                    # Push down Project to the right operand
                    _ = self.inject_projectn_op(
                        join_node_id, right_node_id, right_out_cols, proj_prd)
                    # TODO: Mark as optional predicate instead
                    del proj_node['args']['prd']
                    return 2
                return -4
            if join_type == 'left':
                if proj_prd_in_left and not proj_prd_in_right:
                    # Push down Project to the left operand
                    _ = self.inject_projectn_op(
                        join_node_id, left_node_id, left_out_cols, proj_prd)
                    # TODO: Mark as optional predicate instead
                    del proj_node['args']['prd']
                    return 3
                return -5
            if join_type == 'right':
                if proj_prd_in_right and not proj_prd_in_left:
                    # Push down Project to the right operand
                    _ = self.inject_projectn_op(
                        join_node_id, right_node_id, right_out_cols, proj_prd)
                    # TODO: Mark as optional predicate instead
                    del proj_node['args']['prd']
                    return 4
                return -6
            # No predicate push down for other join types such as (full) 'outer' (join)
            # since it changes the semantics of the original statement.
            return -7
        return -8

    def push_project_under_groupby(self, node_id):
        """push the projection operator under groupby"""
        proj_node, proj_op, _ = self.get_node(node_id)
        agg_id = self.rwr_plan.children(node_id)[0]
        agg_node, agg_op, agg_func = self.get_node(agg_id)
        if proj_op == Operator.PROJECT and agg_op == Operator.AGGF:
            try:
                optional = proj_node['args']['optional']
            except KeyError:
                pass
            else:
                if 'prd' in optional:
                    # This is a redundant predicate. The predicate must be already pushed down
                    return -1
            # Whitelist the aggregate functions we allow for this rewrite
            if agg_func not in [Function.COUNT, Function.SUM, Function.MIN, Function.MAX, Function.SIZE]:
                return -1
            try:
                agg_axis = agg_node['args']['axis']
            except KeyError:
                # Default axis is 0 or index
                # We can rewrite when the axis=0
                pass
            else:
                if agg_axis not in (0, 'index'):
                    return -1
            gb_id = self.rwr_plan.children(agg_id)[0]
            gb_node, _, _ = self.get_node(gb_id)
            try:
                proj_prd = proj_node['args']['prd']
            except KeyError:
                return -2
            try:
                groupby_cols = gb_node['args']['by']
            except KeyError:
                return -3

            # as_index must be present and set to False. When it's not set, the default is True.
            try:
                as_index = gb_node['args']['as_index']
            except KeyError:
                return -4
            if as_index:
                return -4

            # Collecting the columns in the predicate in PROJECT
            # The predicate is stored in post-order form.
            # We assume it's a comparison op in the form of <compop> <left-opnd> <right-opnd>
            assert len(proj_prd) == 3
            proj_prd_cols = []
            if isinstance(proj_prd[1], str):
                proj_prd_cols.append(proj_prd[1])
            if isinstance(proj_prd[2], str):
                proj_prd_cols.append(proj_prd[2])

            # Column(s) in the predicate must be in the group by columns.
            if any(col not in groupby_cols for col in proj_prd_cols):
                return -5

            # All conditions are met, start rewriting
            _ = self.inject_projectn_op(gb_id, pred=proj_prd.copy())
            # Mark the original PROJECT as an optional node
            try:
                _ = proj_node['args']['optional']
            except KeyError:
                proj_node['args']['optional'] = ['prd']
            else:
                proj_node['args']['optional'].append('prd')
            return 1
        return -1

    def bottom_up_rewrite(self):
        """bottom up rewrite the plan"""
        # A collection of rewrite rules that work by traversing post-order
        # These rules do not change the structure of the input plan
        rule_list = [
            (Operator.SOURCE, self.leaf_node_stats),
            (Operator.GROUPBY, self.infer_groupby_cardinality),
            (Operator.AGGF, self.infer_aggfn_cardinality),
        ]

        node_ids = self.rwr_plan.postorder_nodes(self.rwr_root)
        for node_id in node_ids:
            _, cur_op, _ = self.get_node(node_id)
            for rule in rule_list:
                target_op = rule[0]
                func = rule[1]
                # Run the rewrite rule on the current node
                if target_op == cur_op and not self.ws.is_disable("rwr", cur_op, func.__name__):
                    _ = func(node_id)

        return self.rwr_plan

    def fire_rewrite_rules(self, rule_list, traversal_func):
        """fire rewrite rules"""
        fixed_point = False
        while not fixed_point:
            node_ids = traversal_func(self.rwr_root)
            fixed_point = True
            for node_id in node_ids:
                _, cur_op, _ = self.get_node(node_id)
                for rule in rule_list:
                    target_op = rule[0]
                    func = rule[1]
                    if (target_op == cur_op
                            and not self.ws.is_disable("rwr", cur_op, func.__name__)) or target_op is None:
                        # Run the rewrite on the current node
                        result = func(node_id)
                        if result <= 0:
                            continue
                        # This rule rewrites something, reset the top-down traversal
                        # by breaking the for-loop to re-collect the nodes in the new plan
                        if self.debug:
                            print(f"=== After rewriting node {node_id}:{cur_op} " +
                                  f"with rule {func.__name__} at code point {result} ===")
                            print(self.rwr_plan.print(self.rwr_root,
                                                      self.pr_stats, self.pr_details))
                        fixed_point = False
                        break
                if not fixed_point:
                    # re-run all the rules from the top of the plan
                    break

    def top_down_rewrite(self):
        """top down rewrite the plan"""
        # A collection of rewrite rules that work by traversing depth-first-search
        # These rules may change the structure of the input plan
        rule_list = [
            (Operator.SELECT, self.merge_sel_proj),
            (Operator.PROJECT, self.merge_sel_proj),
            (Operator.PROJECT, self.push_project_under_join),
            (Operator.PROJECT, self.push_project_under_groupby),
        ]
        self.fire_rewrite_rules(rule_list, self.rwr_plan.preorder_nodes)
        return self.rwr_plan

    def run(self):
        """run the rewrite process"""
        # Work on a bottom-up analysis and rewrite
        self.bottom_up_rewrite()
        if self.debug:
            print("=== Logical Plan (after bottom-up Rewrite) ===")
            print(self.rwr_plan.print(self.rwr_root,
                                      self.pr_stats, self.pr_details))

        self.top_down_rewrite()
        if self.debug:
            print("=== Logical Plan (after top-down Rewrite) ===")
            print(self.rwr_plan.print(self.rwr_root,
                                      self.pr_stats, self.pr_details))
        return self.rwr_plan
