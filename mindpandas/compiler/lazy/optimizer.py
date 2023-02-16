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
This module defines an optimizer for lazy mode graph
"""
import math
import mindpandas as mpd
from .graph import LogicalPlan, PhysicalPlan
from .statistics import Statistics
from .query_plan import Function, Operator, fn_op_map


class Optimizer:
    """
    Optimizer for lazy mode graph
    """

    def __init__(self, workspace, plan, debug, pr_stats, pr_details):
        self.ws = workspace
        self.debug = debug
        self.pr_stats = pr_stats
        self.pr_details = pr_details
        # Create a new logical plan
        self.phys_plan = LogicalPlan()

        in_graph = plan.graph
        in_root = plan.root
        dfs_nodes = plan.preorder_nodes(plan.root)
        old_new_map = dict()
        # walk DFS
        for node_id in dfs_nodes:
            node = in_graph.nodes[node_id]
            new_node_id = self.phys_plan.clone_node(node)
            # Add statistics object to each new node
            self.add_statistics_object(new_node_id)
            res = plan.result(node_id)
            if res is not None:
                self.phys_plan.cache(new_node_id, res)

            old_new_map[node_id] = new_node_id
            if node_id == in_root:
                self.phys_plan.root = new_node_id

        for node_id in dfs_nodes:
            edges = plan.edges(node_id)
            for edge in edges:
                new_src_node = old_new_map[edge[0]]
                new_dest_node = old_new_map[edge[1]]
                self.phys_plan.add_edge(new_src_node, new_dest_node)

        # self.phys_graph = self.phys_plan.graph
        # self.phys_root = self.phys_plan.root
        if self.debug:
            print("=== Logical Plan (entering Rewrite) ===")
            print(self.phys_plan.print(self.phys_plan.root,
                                       self.pr_stats, self.pr_details))

    def get_node(self, node_id):
        node = self.phys_plan.graph.nodes[node_id]
        op = node['name']
        func = node['func']
        return node, op, func

    def add_statistics_object(self, node_id):
        if self.phys_plan.stats(node_id) is None:
            self.phys_plan.set_stats(node_id, Statistics())

    def infer_groupby_cardinality(self, node_id):
        """
        Function to infer groupby cardinality
        """
        child_ids = self.phys_plan.children(node_id)
        assert len(child_ids) == 1
        child_id = child_ids[0]
        child_stats = self.phys_plan.stats(child_id)
        assert child_stats is not None
        stats = Statistics()
        if child_stats.ncolumns[0] is not math.nan:
            stats.infer(ncolumns=child_stats.ncolumns[0])
        self.phys_plan.set_stats(node_id, stats)
        return 1

    def infer_aggfn_cardinality(self, node_id):
        """
        Function to infer aggeragation function cardinality.
        """
        cur_node, _, _ = self.get_node(node_id)
        child_ids = self.phys_plan.children(node_id)
        assert len(child_ids) == 1
        child_id = child_ids[0]
        _, _, child_func = self.get_node(child_id)
        child_stats = self.phys_plan.stats(child_id)
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
        self.phys_plan.set_stats(node_id, stats)
        return 1

    def leaf_node_stats(self, node_id):
        """
        Set leaf node stats
        """
        stats = self.phys_plan.stats(node_id)
        if stats.ncolumns[0] is math.nan and stats.nrows[0] is math.nan:
            # If the leaf node has no statistics, do an estimate
            stats = Statistics()
            # Just set some fake numbers
            stats.estimate(nrows=101, ncolumns=8)
            self.phys_plan.set_stats(node_id, stats)
            return 1
        return -1

    def merge_sel_proj(self, node_id):
        """
        Merge select and project
        """
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

            operands = self.phys_plan.children(pred_id)
            lhs_id = operands[0]
            lhs_node, _, lhs_func = self.get_node(lhs_id)
            if lhs_func != Function.PROJECT1:
                matched = 0  # failed
                return matched, lhs, op, const

            common_child_ids = self.phys_plan.children(lhs_id)
            common_child_id = common_child_ids[0]
            if first_op == Operator.PROJECT:
                # merge the select with project
                proj_child_ids = self.phys_plan.children(first_id)
                proj_child_id = proj_child_ids[0]
                if common_child_id == proj_child_id:
                    # +- [1] Select(select)
                    # |  +- [2] Project(project-n) or Project(project-1)
                    # |  |  *- [3] <node>
                    # |  +- [4] ScalarFn(compop)
                    # |  |  +- [5] Project(project-1)
                    # |  |  |  *- [3] <same-node-as-above>
                    matched = 1  # found pattern 1
                    try:
                        _ = self.phys_plan.graph.nodes[first_id]['args']['prd']
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
                # +- [1] Select(select)
                # |  *- [2] <node>
                # |  +- [3] ScalarFn(compop)
                # |  |  +- [4] Project(project-1)
                # |  |  |  *- [2] <same-node-as-above>
                matched = 2  # found pattern 2
            else:
                matched = 0  # failed
                return matched, lhs, op, const

            lhs = lhs_node['args']['column']
            op = pred_node['args']['op']
            const = pred_node['args']['other']
            return matched, lhs, op, const

        _, cur_op, _ = self.get_node(node_id)
        child_ids = self.phys_plan.children(node_id)
        child_id = child_ids[0]
        _, child_op, _ = self.get_node(child_id)
        if cur_op == Operator.SELECT:
            if child_op != Operator.SELECT:
                matched, lhs, op, const = check_sel_pattern(child_ids)
                if matched == 1:
                    # Input:
                    # +- [1] Select(select)
                    # |  +- [2] Project(project-n) or Project(project-1)
                    # |  |  *- [3] <node>
                    # |  +- [4] ScalarFn(compop)
                    # |  |  +- [5] Project(project-1)
                    # |  |  |  *- [3] <same-node-as-above>
                    select_id = node_id
                    project_id = child_id
                    self.phys_plan.graph.remove_edge(select_id, project_id)
                    parents = self.phys_plan.parents(select_id)
                    for p in parents:
                        self.phys_plan.graph.remove_edge(p, select_id)
                        self.phys_plan.graph.add_edge(p, project_id)
                    if self.phys_plan.root == select_id:
                        # Re-position the root to the Project
                        self.phys_plan.root = project_id

                    # Output (after rewrite):
                    # +- [2] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                    # |  +- [3] <node>

                    # If [1] Select is the root node of the plan, we need to set the [2] Project as the new root.
                    # Add the pred to the Project
                    pred_expr = [op, lhs, const]  # stored in post-order
                    project_node, _, _ = self.get_node(project_id)
                    project_node['args']['prd'] = pred_expr
                    return 1
                if matched == 2:
                    # +- [1] Select(select)
                    # |  *- [2] <node>
                    # |  +- [3] ScalarFn(compop)
                    # |  |  +- [4] Project(project-1)
                    # |  |  |  *- [2] <same-node-as-above>
                    select_id = node_id
                    common_node_id = child_id
                    self.phys_plan.graph.remove_edge(select_id, common_node_id)
                    proj_n_id = self.phys_plan.add_1op_node(name=fn_op_map[Function.PROJECTN], fn=Function.PROJECTN,
                                                            child=common_node_id)
                    parents = self.phys_plan.parents(select_id)
                    for p in parents:
                        self.phys_plan.graph.remove_edge(p, select_id)
                        self.phys_plan.graph.add_edge(p, proj_n_id)
                    if self.phys_plan.root == select_id:
                        # Re-position the root to the Project
                        self.phys_plan.root = self.phys_plan.root = proj_n_id

                    # Output (after rewrite):
                    # +- [5] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                    # |  +- [3] <node>

                    # Note [5] is a new node. If [1] Select is the root node of the plan,
                    # we need to set the [5] Project as the new root.
                    # Add the pred to the Project
                    pred_expr = [op, lhs, const]  # stored in post-order
                    project_node, _, _ = self.get_node(proj_n_id)
                    project_node['args']['prd'] = pred_expr
                    return 1
                # no rewrite
                return -2
            # Future extension to merge the 2 selects
            return -1
        if cur_op == Operator.PROJECT:
            try:
                _ = self.phys_plan.graph.nodes[node_id]['args']['prd']
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
                sel_child_ids = self.phys_plan.children(select_id)
                matched, lhs, op, const = check_sel_pattern(sel_child_ids)
                if matched == 2:
                    # Input:
                    # +- [1] Project(project-n) or Project(project-1)
                    # |  +- [2] Select(select)
                    # |  |  *- [3] <node>
                    # |  |  +- [4] ScalarFn(compop)
                    # |  |  |  +- [5] Project(project-1)
                    # |  |  |  |  *- [3] <same-node-as-above>
                    src_id = sel_child_ids[0]
                    self.phys_plan.graph.remove_edge(project_id, select_id)
                    self.phys_plan.graph.add_edge(project_id, src_id)
                    # Output (after rewrite):
                    # +- [1] Project(project-n) or Project(project-1) with predicate <lhs><op><const>
                    # |  +- [3] <node>
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

    def replace_join(self, node_id):
        """
        Function to replace join.
        """
        # TODO, check order before tuning JOIN into MAP2
        _, _, join_fn = self.get_node(node_id)
        node = self.phys_plan.graph.nodes[node_id]

        children = self.phys_plan.children(node_id)
        if len(children) == 1:
            # running against join value
            child_id = self.phys_plan.children(node_id)[0]
            map_op_id = self.phys_plan.add_1op_node(
                name=Operator.MAP1, fn=join_fn, child=child_id, **node['args'])
        elif len(children) == 2:
            main_child_id = self.phys_plan.children(node_id)[0]
            other_child_id = self.phys_plan.children(node_id)[1]
            map_op_id = self.phys_plan.add_2op_node(name=Operator.MAP2, fn=join_fn, right_child=other_child_id,
                                                    left_child=main_child_id, **node['args'])
        else:
            assert 0  # not supported - crash out

        for child_id in children:
            self.phys_plan.remove_edge(node_id, child_id)

        for parent in self.phys_plan.parents(node_id):
            self.phys_plan.add_edge(parent, map_op_id)
            self.phys_plan.remove_edge(parent, node_id)

        if self.phys_plan.root == node_id:
            self.phys_plan.root = map_op_id

        return 0

    def replace_aggr(self, node_id):
        """
        Function to replace aggr.
        """
        _, _, aggr_fn = self.get_node(node_id)
        node = self.phys_plan.graph.nodes[node_id]
        # Median is only one stage reduce
        if aggr_fn in (Function.MEDIAN, Function.COUNT):
            two_stage = False
        else:
            two_stage = True

        if two_stage:
            _ = str(aggr_fn) + '_reduce'

            arg = node['args']
            child_id = self.phys_plan.children(node_id)[0]
            new_map_id = self.phys_plan.add_1op_node(
                name=Operator.MAP1, fn=aggr_fn, child=child_id, **node['args'])
            # TODO mean should be same as other ops like min, max
            arg['concat_axis'] = arg.get('axis', None) ^ 1
            new_reduce_id = self.phys_plan.add_1op_node(
                name=Operator.REDUCEPARTITIONS, fn=aggr_fn, child=new_map_id, **node['args'])

            self.phys_plan.remove_edge(node_id, child_id)

            for parent in self.phys_plan.parents(node_id):
                self.phys_plan.add_edge(parent, new_reduce_id)
                self.phys_plan.remove_edge(parent, node_id)
            self.phys_plan.graph.remove_node(node_id)
        else:
            # single reduce
            child_id = self.phys_plan.children(node_id)[0]
            new_reduce_id = self.phys_plan.add_1op_node(
                name=Operator.REDUCEPARTITIONS, fn=aggr_fn, child=child_id, **node['args'])

            self.phys_plan.remove_edge(node_id, child_id)

            for parent in self.phys_plan.parents(node_id):
                self.phys_plan.add_edge(parent, new_reduce_id)
                self.phys_plan.remove_edge(parent, node_id)
        # FUTURE: 3 stage reduce.. one in region.. one in task chain... on within a map

        if self.phys_plan.root == node_id:
            self.phys_plan.root = new_reduce_id

        return 0  # return 0 so we don't loop... maybe need to change op name

    def replace_reduce(self, node_id):
        """
        Function to replace reduce.
        """
        _, _, reduce_fn = self.get_node(node_id)
        node = self.phys_plan.graph.nodes[node_id]
        two_stage = True
        if two_stage:
            child_id = self.phys_plan.children(node_id)[0]
            new_map_id = self.phys_plan.add_1op_node(
                name=Operator.MAP1, fn=reduce_fn, child=child_id, **node['args'])
            new_reduce_id = self.phys_plan.add_1op_node(
                name=Operator.REDUCEPARTITIONS, fn=reduce_fn, child=new_map_id, **node['args'])

            self.phys_plan.remove_edge(node_id, child_id)

            for parent in self.phys_plan.parents(node_id):
                self.phys_plan.add_edge(parent, new_reduce_id)
                self.phys_plan.remove_edge(parent, node_id)
            self.phys_plan.graph.remove_node(node_id)
        else:
            # single reduce
            child_id = self.phys_plan.children(node_id)[0]
            new_reduce_id = self.phys_plan.add_1op_node(
                name=Operator.REDUCEPARTITIONS, fn=reduce_fn, child=child_id, **node['args'])

            self.phys_plan.remove_edge(node_id, child_id)

            for parent in self.phys_plan.parents(node_id):
                self.phys_plan.add_edge(parent, new_reduce_id)
                self.phys_plan.remove_edge(parent, node_id)

        if self.phys_plan.root == node_id:
            self.phys_plan.root = new_reduce_id

        return 0  # return 0 so we don't loop... maybe need to change op name

    def replace_project(self, node_id):
        proj_node, _, _ = self.get_node(node_id)
        # change operator type to a MAP
        proj_node['name'] = Operator.MAP1
        return 0

    def inject_op(self, parent_id, child_id, func_type=Function.PROJECTN):
        """
        Inject a new operator between the parent and the child
        """
        new_node_id = self.phys_plan.add_1op_node(
            name=fn_op_map[func_type], fn=func_type, child=child_id)
        children_ids = self.phys_plan.children(parent_id)
        found = False
        # Maintain the correct order of the children of the parent
        for node_id in children_ids:
            if found or node_id == child_id:
                self.phys_plan.graph.remove_edge(parent_id, node_id)
                found = True
        assert found

        found = False
        for node_id in children_ids:
            if node_id == child_id:
                self.phys_plan.graph.add_edge(parent_id, new_node_id)
                found = True
            elif found:
                self.phys_plan.graph.add_edge(parent_id, node_id)
        return new_node_id

    def inject_projectn_op(self, parent_id, child_id=None, out_columns=None, pred=None):
        """
        Inject PROJECTN op.
        """
        if child_id is None:
            # insert above the only child of the parent
            children = self.phys_plan.children(parent_id)
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
        join_node_id = self.phys_plan.children(node_id)[0]
        join_node, join_op, _ = self.get_node(join_node_id)
        if proj_op == Operator.PROJECT and join_op == Operator.JOIN:
            try:
                proj_prd = proj_node['args']['prd']
            except KeyError:
                return -1

            join_type = join_node['args']['how']
            try:
                join_cols = join_node['args']['on']
            except KeyError:
                join_cols = None
            if join_cols is not None:
                _ = _ = join_cols
            else:
                try:
                    _ = join_node['args']['left_on']
                except KeyError:
                    _ = None
                try:
                    _ = join_node['args']['right_on']
                except KeyError:
                    _ = None

            grandchildren = self.phys_plan.children(join_node_id)
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
        """
        Push project under groupby.
        """
        proj_node, proj_op, _ = self.get_node(node_id)
        agg_id = self.phys_plan.children(node_id)[0]
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
            gb_id = self.phys_plan.children(agg_id)[0]
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
        """
        A collection of rewrite rules that work by traversing post-order
        These rules do not change the structure of the input plan
        """
        rule_list = [
            (Operator.SOURCE, self.leaf_node_stats),
            (Operator.GROUPBY, self.infer_groupby_cardinality),
            (Operator.AGGF, self.infer_aggfn_cardinality),
        ]

        node_ids = self.phys_plan.postorder_nodes(self.phys_plan.root)
        for node_id in node_ids:
            _, cur_op, _ = self.get_node(node_id)
            for rule in rule_list:
                target_op = rule[0]
                func = rule[1]
                # Run the rewrite rule on the current node
                if target_op == cur_op and not self.ws.is_disable("rwr", cur_op, func.__name__):
                    _ = func(node_id)

        return self.phys_plan

    def fire_rewrite_rules(self, rule_list, traversal_func):
        """
        Fire rewrite rules
        """
        fixed_point = False
        while not fixed_point:
            node_ids = traversal_func(self.phys_plan.root)
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
                        # by breaking the for-loop to re-collect the nodes in the new plan.
                        if self.debug:
                            print(f"=== After rewriting node {node_id}:{cur_op} " +
                                  f"with rule {func.__name__} at code point {result} ===")
                            print(self.phys_plan.print(
                                self.phys_plan.root, self.pr_stats, self.pr_details))
                        fixed_point = False
                        break
                if not fixed_point:
                    # re-run all the rules from the top of the plan
                    break

    def infer_partitions(self, node_id):
        """
        Infer partitions
        """
        current_stats = self.phys_plan.stats(node_id)
        child_ids = self.phys_plan.children(node_id)
        child_stats = None

        if len(child_ids) == 1:
            child_stats = self.phys_plan.stats(child_ids[0])

        elif len(child_ids) > 1:
            # TODO: search for largest input
            child_stats = Statistics()
            for child in child_ids:
                new_child_stats = self.phys_plan.stats(child)
                if new_child_stats.shape[0][0] > child_stats.shape[0][0]:
                    child_stats = new_child_stats
        # Get axis from list
        node, op, _ = self.get_node(node_id)

        if current_stats is None:
            current_stats = Statistics()
            current_stats.infer_from_other(child_stats)
        axis = None
        if op in [Operator.REDUCEPARTITIONS]:
            try:
                axis = node['args']['axis']
            except (KeyError, ValueError, TypeError):
                axis = 0
        if child_stats is not None:
            current_stats.infer(shape=child_stats.shape[0], axis=axis)
        else:
            current_stats.infer(axis=axis)
        self.phys_plan.set_stats(node_id, current_stats)
        return 0

    def verify_no_reduce(self, node_id):
        # Verify we don't have Logical Reduce
        _, op, _ = self.get_node(node_id)
        if op in [Operator.REDUCE]:
            raise RuntimeError("Logical REDUCE is found in Physical Mapping")
        return 0

    def physical_mapping(self):
        """
        A collection of rewrite rules that work by traversing depth-first-search
        These rules may change the structure of the input plan
        """
        rule_list_top = [
            (Operator.JOIN, self.replace_join),
            (Operator.AGGF, self.replace_aggr),
            (Operator.REDUCE, self.replace_reduce),
            (Operator.REDUCEPARTITIONS, self.replace_reduce),
            (Operator.PROJECT, self.replace_project),
        ]
        self.fire_rewrite_rules(rule_list_top, self.phys_plan.preorder_nodes)
        rule_list_bottom = [
            (None, self.infer_partitions),
            (None, self.verify_no_reduce),
        ]
        self.fire_rewrite_rules(
            rule_list_bottom, self.phys_plan.postorder_nodes)
        return self.phys_plan

    def _check_prev_taskchain_children(self, child_node_id):
        """
        Check if a child TaskChain root deals with more than 1 input
        If more than 1 input - do not merge with for now
        """
        # source_count = 0
        # child_node, _, _ = self.get_node(child_node_id)
        # graph = child_node['args']['_graph'].graph.nodes
        # for node in graph:
        #     if (self.get_node(node)[1] == Operator.SOURCE):
        #         source_count += 1
        # if (source_count <= 1):
        #     return True
        # return False
        _, _, _ = self.get_node(child_node_id)
        return True  # return default - above can be used for debugging when needed

    def task_chain_merge_check(self, node_id):
        """
        Check task chain merge
        """
        _, op, _ = self.get_node(node_id)
        child_ids = self.phys_plan.children(node_id)
        parent_ids = self.phys_plan.parents(node_id)
        # rule 1: one child is task chain
        cond1 = False

        # if op == Operator.MAP1 and len(child_ids) == 1:
        #     child_node, child_op, child_fn = self.get_node(child_ids[0])
        #     if child_fn == Function.TASKCHAIN:
        #         cond1 = True

        if op in (Operator.MAP1, Operator.PROJECT, Operator.AUX, Operator.MAP2):
            for child_id in child_ids:
                _, _, child_fn = self.get_node(child_id)
                if child_fn == Function.TASKCHAIN:
                    cond1 = True
        # rule2: so far we support only up to 2 parents
        cond2 = len(parent_ids) <= 2

        # rule3, limit size of task chains
        dfs_nodes = self.phys_plan.postorder_nodes(self.phys_plan.root)
        cond3 = len(dfs_nodes) <= mpd.config.ops_per_chain

        # rule4, split at cse... TODO in future to have all cse in same chain
        # cond4 = False
        # is_cse = self.phys_plan.cse(node_id)
        # child_cse = False
        # for i in range(len(child_ids)):
        #     child_cse |= self.phys_plan.cse(child_ids[i])
        # cond4 = (is_cse and child_cse) or (not is_cse and not child_cse)

        # return cond1 and cond2 and cond3 and cond4
        return cond1 and cond2 and cond3

    def task_chain_new_check(self, node_id):
        _, op, _ = self.get_node(node_id)
        return op not in (Operator.SOURCE, Operator.SOURCEREP, Operator.AUX, Operator.VIEW)

    def visit(self, node_id, task_chain, breakpoints, reduce_nodes, visited):
        """
        Visits a node(node_id) and all its parents and children recursively.
        :param node_id: the id of the node to be visited
        :param task_chain: a list containing the node_ids for the current task chain
        :param breakpoints: a list containing the node_ids for the breakpoints.
            definition: A breakpoint is the parent of a reduce or source node.
        :param reduce_nodes: a list containing the node_ids for the reduce nodes.
        :param visited: a list containing the node_ids visited so far to avoid revisiting.

        "task_chain, breakpoints, reduce_nodes, visited" gets updated during each visit.
        """

        phys_plan = self.phys_plan
        node = phys_plan.graph.nodes[node_id]
        op = node['name']
        if node_id in visited:
            print('node %d is visited' % node_id)
        else:
            # set node status as visited
            visited.append(node_id)
            if op in [Operator.SOURCE, Operator.SOURCEREP, Operator.REDUCE,
                      Operator.REDUCEPARTITIONS, Operator.REDUCEBYKEY, Operator.VIEW]:
                # this node is reduce or a source
                children = phys_plan.children(node_id)
                if children:
                    reduce_nodes[node_id] = phys_plan.children(node_id)
                return
            # this node is map like
            task_chain.append(node_id)

            child_ids = phys_plan.children(node_id)
            for child_id in child_ids:
                is_breakpoint = False
                node = phys_plan.graph.nodes[child_id]
                op = node['name']
                if op in [Operator.SOURCE, Operator.SOURCEREP, Operator.REDUCE, Operator.REDUCEPARTITIONS,
                          Operator.REDUCEBYKEY, Operator.VIEW]:
                    is_breakpoint = True
                if is_breakpoint:
                    # the visited node is a breakpoint
                    if node_id in breakpoints:
                        breakpoints[node_id].append(child_id)
                    else:
                        breakpoints[node_id] = [child_id]

                if not child_id in visited:
                    # visit the child of the current node
                    self.visit(child_id, task_chain, breakpoints,
                               reduce_nodes, visited)

            parent_ids = phys_plan.parents(node_id)
            for parent_id in parent_ids:
                node = phys_plan.graph.nodes[parent_id]
                if not parent_id in visited:
                    # visit the parent of the current node
                    self.visit(parent_id, task_chain,
                               breakpoints, reduce_nodes, visited)

    def set_root_destination(self, task_graph, chain_nodes, reduce_nodes, map_node_taskchain, task_graph_dict):
        """
        Sets the destination and roots for the task graph.

        :param task_graph: the input task_graph
        :param chain_nodes: the nodes inside the task_graph
        :param reduce_nodes: a map from a reduce/breakpoint node to its children
        :param map_node_taskchain: a map from a node_id to its chain_id
        :param task_graph_dict: a map from a chain_ids to actual id of a task chain
        """
        # each task graph can have different number of roots and destination
        # a root is connected to a reduce/breakpoint node and has no parent in the task chain
        # a destination is connected to a reduce/breakpoint node
        for node_id in reduce_nodes:
            nominated_roots = reduce_nodes[node_id]
            for root_id in nominated_roots:
                if root_id in chain_nodes:
                    root_parents = task_graph.parents(root_id)
                    if not root_parents:
                        # this is a root
                        if root_id not in task_graph.root_list:
                            task_graph.root_list.append(root_id)
                        # get destination
                        chain_id = map_node_taskchain[node_id]
                        dest_id = task_graph_dict[chain_id]
                        # add it to the result list
                    if root_id not in task_graph.dest_map:
                        task_graph.dest_map[root_id] = [dest_id]
                    else:
                        task_graph.dest_map[root_id].append(dest_id)

        # in case of the root task_graph:
        if not task_graph.root_list:
            task_graph.root_list.append(chain_nodes[0])
        if not task_graph.dest_map:
            # the root task graph has no destination
            task_graph.dest_map[chain_nodes[0]] = []

    def create_task_chain_optimized_v1(self):
        """
        Creates the task chains.
        First, it traverse the graph and visits each node and its children and parents recursively.
        Then it creates the task graphs and connects them.
        Finally, it cleans the physical plan by removing the original nodes.
        """
        phys_plan = self.phys_plan
        bfs_nodes = phys_plan.bfs_nodes(phys_plan.root)
        visited = []
        task_chain_nodes = {}  # contains the node ids for each task chain
        # contains the breakpoints(a node that has reduce or source as a child)
        breakpoints = {}
        reduce_nodes = {}  # contains the reduce nodes

        task_chain_id = 0
        for node_id in bfs_nodes:
            task_chain = []
            reduce_nodes_ = {}
            # visit the node and its children and parents recursively
            self.visit(node_id, task_chain, breakpoints,
                       reduce_nodes_, visited)
            if task_chain:
                task_chain_nodes[task_chain_id] = task_chain
                task_chain_id += 1

            # add the reduce nodes to separate task chains
            for reduce_node_id in reduce_nodes_:
                task_chain_nodes[task_chain_id] = [reduce_node_id]
                task_chain_id += 1
            reduce_nodes.update(reduce_nodes_)

        # create a map from nodes to task_chain_id
        # this is useful in the next step when creating the task_chain_nodes
        map_node_taskchain = {}
        for i in task_chain_nodes:
            for node_id in task_chain_nodes[i]:
                map_node_taskchain[node_id] = i

        #### creating the task graphs ####

        # step 1: add the reference nodes
        # right now we add a ref node for each breakpoint and reduce node
        all_nodes = breakpoints.copy()
        all_nodes.update(reduce_nodes)

        for task_chain_id in task_chain_nodes:
            chain_nodes = task_chain_nodes[task_chain_id]
            if phys_plan.graph.nodes[chain_nodes[0]]['name'] == Operator.VIEW:
                continue
            offset_map = {}
            offset = 0
            for node_id in chain_nodes:
                if node_id in all_nodes:
                    # node_id is a either conntected to a source or reduce
                    for source_id in all_nodes[node_id]:
                        node, _, _ = self.get_node(source_id)
                        # make a reference node
                        if source_id in offset_map:
                            source_node_id = self.phys_plan.add_node(
                                name=Operator.SOURCE, fn=Function.NODE,
                                stats=node['stats'], _offset=offset_map[source_id], _id=source_id)
                        else:
                            source_node_id = self.phys_plan.add_node(
                                name=Operator.SOURCE, fn=Function.NODE, stats=node['stats'],
                                _offset=offset, _id=source_id)
                            offset_map[source_id] = offset
                            offset += 1

                        self.phys_plan.add_edge(node_id, source_node_id)
                        # update the task chain nodes by adding the reference node
                        task_chain_nodes[task_chain_id].append(source_node_id)

        # step 2:
        # create the task graphs for map operations
        task_graph_dict = {}
        for chain_id in task_chain_nodes:
            print('creating task chain %d' % chain_id)
            chain_nodes = task_chain_nodes[chain_id]
            # chain_nodes[0] is the root of the task chain
            # in order to find the proper name for a task graph,
            # we look into its node and check the number of source nodes(breakpoints) it is connected to
            num_sources = 0
            all_sources = []
            kwargs = {}
            kwargs['colocation'] = []

            for node_id in chain_nodes:
                if node_id in breakpoints:
                    all_sources = all_sources + \
                        list(set(breakpoints[node_id]) - set(all_sources))

                # we set the kwargs here too
                node = self.phys_plan.graph.nodes[node_id]
                args = node['args']
                if 'parallelism' in args:
                    kwargs['parallelism'] = args['parallelism']

                if node['name'] == Operator.REDUCEPARTITIONS:
                    if 'axis' in node['args'].keys():
                        kwargs['axis'] = node['args']['axis']
                # get the "columns" in the task chain (used for repartition later)
                # TODO: avoid duplicates here
                if 'column' in args:
                    kwargs['colocation'].append(args['column'])
                if 'columns' in args:
                    kwargs['colocation'].append(args['columns'])
                if 'by' in args:
                    kwargs['colocation'].append(args['by'])
                if 'concat_axis' in args:
                    kwargs['concat_axis'] = args.pop('concat_axis', 0)
            num_sources = len(all_sources)
            assert num_sources < 3  # we only support Map1 and Map2

            # TODO: set the stats properly
            cur_node, cur_op, _ = self.get_node(chain_nodes[0])
            if num_sources > 1:
                # any node with more than 1 input is a map2?
                # we don't have reduce with two inputs so far
                cur_op = Operator.MAP2

            if phys_plan.graph.nodes[chain_nodes[0]]['name'] == Operator.VIEW:
                task_graph_dict[chain_id] = chain_nodes[0]
            else:
                task_graph = PhysicalPlan()
                task_graph.copy_graph(self.phys_plan, chain_nodes)

                # set the list of roots and destination for the task graph:
                self.set_root_destination(
                    task_graph, chain_nodes, all_nodes, map_node_taskchain, task_graph_dict)
                print(task_graph.print(task_graph.root, False, False))

                new_node_id = self.phys_plan.add_node(
                    name=cur_op, fn=Function.TASKCHAIN, _graph=task_graph, stats=cur_node['stats'], **kwargs)
                task_graph_dict[chain_id] = new_node_id
                if chain_nodes[0] == self.phys_plan.root:
                    self.phys_plan.root = new_node_id

        # adding children and parents
        def add_list_of_nodes(input_list):
            # adds the input list(either breakpoints or reduce nodes) to the graph
            visited = {}  # to avoid duplicate adding
            for node_id in input_list:
                for source_id in input_list[node_id]:
                    task_graph_id = map_node_taskchain[node_id]
                    task_graph_node_source = source_id
                    if source_id in map_node_taskchain:
                        task_graph_source_id = map_node_taskchain[source_id]
                        task_graph_node_source = task_graph_dict[task_graph_source_id]
                    task_graph_node = task_graph_dict[task_graph_id]

                    # avoid duplicate adding
                    if not (task_graph_id in visited and visited[task_graph_id] == task_graph_node_source):
                        self.phys_plan.add_edge(
                            task_graph_node, task_graph_node_source)
                        if task_graph_id in visited:
                            visited[task_graph_id].append(
                                task_graph_node_source)
                        else:
                            visited[task_graph_id] = [task_graph_node_source]

        # step 3: add parents and children for each task graph
        add_list_of_nodes(breakpoints)
        add_list_of_nodes(reduce_nodes)

        # clean the chain nodes from graph
        for chain_id in task_chain_nodes:
            chain_nodes = task_chain_nodes[chain_id]
            if phys_plan.graph.nodes[chain_nodes[0]]['name'] == Operator.VIEW:
                continue
            for node in chain_nodes:
                self.phys_plan.graph.remove_node(node)

    def create_task_chain2(self):
        """
        Create task chain.
        """
        dfs_nodes = self.phys_plan.postorder_nodes(self.phys_plan.root)
        # algorithm
        # 1. loop ver all nodes
        #     a. create new task chain if 1) no adjacent task chain 2) multiple children 3) different parallelism
        #     b. mrege into task chain if parallelism same and no condition for new
        #     c. leave it if it is read op
        for node_id in dfs_nodes:
            print(node_id, self.phys_plan.print(
                self.phys_plan.root, self.pr_stats, self.pr_details))

            if self.task_chain_merge_check(node_id):
                child_ids = self.phys_plan.children(node_id)
                node, _, _ = self.get_node(node_id)
                node_args = node['args']
                for child_id in child_ids:
                    child_node, _, child_fn = self.get_node(child_id)
                    if child_fn != Function.TASKCHAIN:
                        continue
                    args = child_node['args']
                    # get the columns, if exists
                    # used for repartion by columns later
                    if 'colocation' not in args:
                        break
                    if 'column' in node_args:
                        item = node_args['column']
                        if not item in child_node['args']['colocation']:
                            child_node['args']['colocation'].append(item)
                    if 'columns' in node_args:
                        item = node_args['columns']
                        if not item in child_node['args']['colocation']:
                            child_node['args']['colocation'].append(item)
                    break
                inner_plan = child_node['args']['_graph']
                self.phys_plan.move_node_to_plan_at_root(node_id, inner_plan)
                # self.phys_plan.delete_node(node_id)

            elif self.task_chain_new_check(node_id):

                # create new TASKCHAIN node with node_id contents
                task_graph = PhysicalPlan()
                chain_nodes = [node_id]
                child_ids = self.phys_plan.children(node_id)
                source_node_id1 = False
                cur_node, cur_op, _ = self.get_node(node_id)

                if len(child_ids) <= 1:
                    node, _, _ = self.get_node(child_ids[0])
                    source_node_id1 = self.phys_plan.add_node(
                        name=Operator.SOURCE, fn=Function.NODE, stats=node['stats'], _offset=0, _id=child_ids[0])
                    self.phys_plan.add_edge(node_id, source_node_id1)
                    chain_nodes.append(source_node_id1)

                elif len(child_ids) == 2:  # need better solution for this
                    node1, _, _ = self.get_node(child_ids[0])  # left - main
                    node2, _, _ = self.get_node(
                        child_ids[1])  # right - secondary
                    source_node_id1 = self.phys_plan.add_node(
                        name=Operator.SOURCE, fn=Function.NODE, stats=node1['stats'], _offset=0, _id=child_ids[0])
                    source_node_id2 = self.phys_plan.add_node(
                        name=Operator.SOURCE, fn=Function.NODE, stats=node2['stats'], _offset=1, _id=child_ids[1])
                    for source_node_id in [source_node_id1, source_node_id2]:
                        self.phys_plan.add_edge(node_id, source_node_id)
                        chain_nodes.append(source_node_id)
                else:
                    print("ERROR child_ids > 2")

                # get first node parallelism
                node, _, _ = self.get_node(node_id)
                args = node['args']
                kwargs = {}
                if 'parallelism' in args:
                    kwargs['parallelism'] = args['parallelism']
                if cur_node['name'] == Operator.REDUCEPARTITIONS:
                    if 'axis' in cur_node['args'].keys():
                        kwargs['axis'] = cur_node['args']['axis']

                # get the "columns" in the task chain (used for repartition later)
                kwargs['colocation'] = []
                if 'column' in args:
                    kwargs['colocation'].append(args['column'])
                if 'columns' in args:
                    kwargs['colocation'].append(args['columns'])

                # OPEN: should we check other nodes here first???
                task_graph.copy_graph(self.phys_plan, chain_nodes)
                new_node_id = self.phys_plan.add_node(
                    name=cur_op, fn=Function.TASKCHAIN, _graph=task_graph, stats=cur_node['stats'], **kwargs)
                # replace current node with task chain node
                if source_node_id1:
                    self.phys_plan.remove_edge(node_id, source_node_id1)
                    if len(child_ids) == 2:
                        self.phys_plan.remove_edge(node_id, source_node_id2)
                self.phys_plan.replace_node(node_id, new_node_id)
            # print("end rewrite2",node_id, self.phys_plan.print(self.phys_plan.root, self.pr_stats, self.pr_details))

            # else do nothing to this node

    def create_repartition(self):
        """
        Replace the current source node with a new source node and replicates
        the paths from source to other maps. Each path is used to process a partition later.

        In the stream mode, the new source node reads chunk of data and divides it into partitions.
        number of partitions is set in config->partition_shape[1]

        """
        in_graph = self.phys_plan.graph
        in_root = self.phys_plan.root
        dfs_nodes = self.phys_plan.preorder_nodes(self.phys_plan.root)
        new_phys_plan = LogicalPlan()
        old_new_map = dict()
        colocation = []
        for node_id in dfs_nodes:
            node = in_graph.nodes[node_id]
            func = node['func']
            name = node['name']
            args = node['args']
            # print('physical graph node id', node_id, ' func', func, ' name', name)

            # replicate the paths containing maps for each partition
            # here we only care about the repartitioning on the columns
            if func == Function.TASKCHAIN:
                if 'colocation' in args:
                    colocation.append(args['colocation'])
                for i in range(mpd.config.partition_shape[1]):
                    new_node_id = new_phys_plan.clone_node(node)
                    if not node_id in old_new_map:
                        old_new_map[node_id] = []
                    old_new_map[node_id].append(new_node_id)
            else:
                # replace the source with new source operation to create partitions later
                if name == Operator.SOURCE:
                    node['name'] = Operator.SOURCEREP

                new_node_id = new_phys_plan.clone_node(node)
                if not node_id in old_new_map:
                    old_new_map[node_id] = []
                old_new_map[node_id].append(new_node_id)

            if node_id == in_root:
                new_phys_plan.root = new_node_id

        # add the edges to the new graph
        for node_id in dfs_nodes:
            edges = self.phys_plan.edges(node_id)
            for edge in edges:
                new_src_nodes = old_new_map[edge[0]]
                new_dest_nodes = old_new_map[edge[1]]
                # add colocation to sourcce nodes:
                for source_id in new_dest_nodes:
                    node = new_phys_plan.graph.nodes[source_id]
                    name = node['name']
                    if name == Operator.SOURCEREP:
                        node['args']['colocation'] = colocation
                for i in range(len(new_src_nodes)):
                    for j in range(len(new_dest_nodes)):
                        new_phys_plan.add_edge(
                            new_src_nodes[i], new_dest_nodes[j])
        self.phys_plan = new_phys_plan

    def create_subgraph(self):
        # A collection of rewrite rules that work by traversing depth-first-search
        # These rules may change the structure of the input plan
        '''
        region_plan = PhysicalPlan()
        region_plan.copy_graph(self.phys_plan)
        new_node_id = self.phys_plan.add_node(name=Operator.REGION, fn=region_plan, stats=None)
        self.phys_plan.root = new_node_id
        '''
        region_plan = PhysicalPlan()
        new_node_id = region_plan.add_node(
            name=Operator.REGION, fn=self.phys_plan, stats=None)
        region_plan.root = new_node_id
        self.phys_plan = region_plan

    def run(self):
        """
        Work on a bottom-up analysis and rewrite.
        """
        self.physical_mapping()
        if self.debug:
            print("=== Physical Plan  ===")
            print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, True))
            # print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, self.pr_details))

        self.create_task_chain_optimized_v1()

        if self.debug:
            print("=== Task chain  Plan  ===")
            # print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, True))
            print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, False))

        # TODO, add back in to support parallel streaming
        # if mpd.config.stream_repartition and mpd.config.partition_shape[1]>1:
        #    self.create_repartition()
        # self.create_task_chain2()

        if self.debug:
            print("=== Task chain  after repartition  ===")
            # print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, True))
            print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, False))

        self.create_subgraph()
        if self.debug:
            print("=== subgraph  Plan  ===")
            print(self.phys_plan.print(self.phys_plan.root, self.pr_stats, True))
        return self.phys_plan
