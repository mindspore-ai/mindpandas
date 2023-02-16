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
Module contains ``CoordinatorActor`` class which provides a single coordinator
actor, and ``BatchCoordinator`` class which executes target functions concurrently
"""
import copy

import mindpandas as mpd
from mindpandas.compiler.lazy.query_plan import Operator, Function
from mindpandas.compiler.function_factory import FunctionFactory as ff
from mindpandas.compiler.query_compiler import QueryCompiler as qc


class CoordinatorActor:
    def __init__(self, plan, debug, pr_stats, pr_details, cfg_concurrency, cfg_runtime='nondist'):
        self.coordinator = BatchCoordinator(plan, debug, pr_stats, pr_details, cfg_concurrency, cfg_runtime)

    def run(self):
        return self.coordinator.run()


class BatchCoordinator:
    """
    BatchCoordinator accesses Eager frames which has partition list of all partitions.
    BatchCoordinator will loop over partitions and execute target function in parallel.
    """
    def __init__(self, plan, debug, pr_stats, pr_details, cfg_concurrency, cfg_runtime):
        # TODO: Does BatchCoordinator need cfg_concurrency and cfg_runtime? if no, then do not add
        self.plan = plan.create_copy()
        # self.plan = copy.deepcopy(plan)
        self.graph = plan.graph
        self.root = plan.root
        self.debug = debug
        self.pr_stats = pr_stats
        self.pr_details = pr_details
        self.cfg_concurrency = cfg_concurrency
        self.cfg_runtime = cfg_runtime

    def run_task_chain(self, node_id, **kwargs):
        """Runs the methods for a given node."""
        task_graph = kwargs['_graph']
        node_id = self.graph.nodes[node_id]['id']
        name = self.graph.nodes[node_id]['name']
        child_ids = self.plan.children(node_id)
        if len(child_ids) == 1:
            input_frame = self.plan.result(child_ids[0])
            fn = ff.task_chain(task_graph, multi_input=False)
        elif len(child_ids) == 2:
            input_frame1 = self.plan.result(child_ids[0])
            input_frame2 = self.plan.result(child_ids[1])
            fn = ff.task_chain(task_graph, multi_input=True)
        else:
            assert 0
        if fn is None:
            raise NotImplementedError
        if name == Operator.REDUCEPARTITIONS:
            args = self.graph.nodes[node_id]['args']
            axis = args.get("axis", 0)
            concat_axis = args.get("concat_axis", None)
            frame = input_frame.backend_frame.reduce(func=fn, axis=axis, concat_axis=concat_axis)
            if frame.shape[0] == 1 or frame.shape[1] == 1:
                return mpd.Series(data=frame)
            return type(input_frame)(data=frame)
        if name == Operator.MAP1:
            return type(input_frame)(data=input_frame.backend_frame.map(fn))
        if name == Operator.MAP2:
            assert len(child_ids) == 2
            return type(input_frame1)(data=input_frame1.backend_frame.injective_map(None,
                                                                                    input_frame2.backend_frame,
                                                                                    fn,
                                                                                    is_scalar=False))
        assert 0
        return None

    def run_read_csv(self, node_id, **kwargs):
        # Input argument 'node_id' is not used.
        # To conform to the convention of the signature of all "run-time" functions,
        # node_id is passed along with the keyword arguments.
        filepath = kwargs.pop('filepath')
        print("node_id, filepath, kwargs: ", node_id, filepath, kwargs)
        result_df = qc.read_csv(filepath, **kwargs)
        return result_df

    def run_reduce_key(self, node_id, **kwargs):
        child_ids = self.plan.children(node_id)
        assert len(child_ids) == 1
        input_frame = self.plan.result(child_ids[0])
        key_fn = kwargs.pop('keyby_fn')
        reduce_fn = kwargs.pop('reduce_fn')
        return qc.reduceByKey(input_frame, key_fn, reduce_fn)

    def run_view(self, node_id, **kwargs):
        child_ids = self.plan.children(node_id)
        assert len(child_ids) == 1
        input_obj = self.plan.result(child_ids[0])
        return qc.view(input_obj, **kwargs)

    def run(self):
        """Executes the plan by calling appropriate methods for each task."""
        try:
            order_of_exec = self.plan.postorder_nodes(self.root)

            for node_id in order_of_exec:
                func = self.graph.nodes[node_id]['func']
                node_id = self.graph.nodes[node_id]['id']
                name = self.graph.nodes[node_id]['name']

                result = self.plan.result(node_id)

                if result is None:
                    print("in lazy distributed mode, executing", func, node_id, self.graph.nodes[node_id]['name'])
                    # This node has not yet processed. Process it.
                    args = self.graph.nodes[node_id]['args']
                    # Preserve the original plan
                    # Make a copy of the plan arguments
                    # TODO: Do we need to preserve the internal rewr plan/opt plan?
                    kwargs = copy.deepcopy(args)

                    if name == Operator.REGION:
                        #TODO: anyway to get rid of the recursive call?
                        temp_exec = BatchCoordinator(func, self.debug, self.pr_stats, self.pr_details,
                                                     self.cfg_concurrency, self.cfg_runtime)
                        result = temp_exec.run()
                        result.node_id = node_id
                        self.plan.cache(node_id, result)
                        #print('end region node_id', node_id,' result', result)
                        continue
                    elif func == Function.TASKCHAIN:
                        func_call = self.run_task_chain
                    elif name == Operator.REDUCEBYKEY:
                        func_call = self.run_task_chain
                    elif name == Operator.VIEW:
                        func_call = self.run_view
                    elif func == Function.READ_CSV:
                        func_call = self.run_read_csv
                    else:
                        raise NotImplementedError("Coordinator: unexpected node ", name)

                    try:
                        result = func_call(node_id, **kwargs)
                    except RuntimeError as e:
                        print("in coordinator hit exception calling func", node_id, result, e)
                    try:
                        result.node_id = node_id
                    except RuntimeError as e:
                        print("in coordinator hit exception getting", e)
                    self.plan.cache(node_id, result)

                if result is None:
                    raise RuntimeError("Result of an intermediate operator is None.")

            final_result = self.plan.result(self.root)
            return final_result
        except RuntimeError as e:
            print("hit exception in Coordinator:", e)
