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
Module contains ``Executor`` class which provides methods to run each class
of operation in the plan
"""
import copy

from ..compiler.query_compiler import QueryCompiler as qc
from ..compiler.lazy.query_plan import Function, Operator
from ..compiler.function_factory import FunctionFactory as ff


class Executor:
    """
    Executor provides methods to execute classes of operations such as
    map, groupby, reduce, etc.
    """
    def __init__(self, plan, multi_input=False, dest=None):
        self.plan = copy.deepcopy(plan)
        self.plan.dest_map = plan.dest_map
        self.plan.root_list = plan.root_list
        self.graph = plan.graph
        self.root = plan.root
        self.dest = dest # in case we need result from a specific node in the task_chain
        # variable to say whether input is a single partition or a list of segeral arguments
        # map and reduce ops will not be multi-input as it inly takes one input
        # join would be multi-input=True
        self.multi_input = multi_input

    def run_read_csv(self, node_id, **kwargs):
        """Executes read_csv and returns result"""
        # Input argument 'node_id' is not used.
        # To conform to the convention of the signature of all "run-time" functions,
        # node_id is passed along with the keyword arguments.
        print("run_read_csv node_id: ", node_id)
        filepath = kwargs.get('filepath')
        result_df = qc.read_csv(filepath, **kwargs)
        return result_df

    def run_groupby(self, node_id, **kwargs):
        """Executes groupby and returns result"""
        child_ids = self.plan.children(node_id)
        assert len(child_ids) == 1
        input_frame = self.plan.result(child_ids[0])
        # from ..groupby import DataFrameGroupBy
        # df = DataFrameGroupBy(input_frame,
        # by=kwargs['by'],
        # axis=kwargs['axis'],
        # level=kwargs['level'],
        # as_index=kwargs['as_index'],
        # sort=kwargs['sort'],
        # group_keys=kwargs['group_keys'],
        # squeeze=kwargs['squeeze'],
        # observed=kwargs['observed'],
        # dropna=kwargs['dropna'],
        # by_names=None)
        # print('df is:', df)
        # import time
        # time.sleep(1)
        # fn = ff.keyby(kwargs['by'])
        # keyed_data = fn(input_frame)
        df = input_frame.groupby(kwargs['by'])
        return df

    def run_map(self, node_id, **kwargs):
        """Executes map and returns result"""
        child_ids = self.plan.children(node_id)
        assert len(child_ids) == 1
        input_frame = self.plan.result(child_ids[0])
        func = self.graph.nodes[node_id]['func']
        if func == Function.UDF:
            fn = kwargs.get('fn')
        else:
            dict_map_exception = {'mean': getattr(ff, 'sum_count')}
            if str(func) == 'sum':
                fn = getattr(ff, 'sum_count')
            else:
                fn = dict_map_exception[str(func)] if str(func) in dict_map_exception else getattr(ff, str(func))
        try:
            if func == Function.MATH:
                # running a math op against a scalar value inside op args
                right_object = kwargs.get('other')
                fn = fn(**kwargs)
                output = fn(input_frame, right_object)
            else:
                fn = fn(**kwargs)
                output = fn(input_frame)
        except RuntimeError as e:
            raise Exception("Hit exception in map", e)
        return output

    def run_map2(self, node_id, **kwargs):
        """Executes two dataframe map and returns result"""
        func = self.graph.nodes[node_id]['func']
        child_ids = self.plan.children(node_id)
        left_dataframe = self.plan.result(child_ids[0])
        right_dataframe = self.plan.result(child_ids[1])
        fn = getattr(ff, str(func))
        init_fn = fn(**kwargs)
        try:
            output = init_fn(left_dataframe, right_dataframe)
        except RuntimeError as e:
            print('Hit exception in map2', e)
        return output

    def run_reduce(self, node_id, **kwargs):
        """Executes reduce and returns result"""
        child_ids = self.plan.children(node_id)
        assert len(child_ids) == 1
        input_frame = self.plan.result(child_ids[0])
        func = self.graph.nodes[node_id]['func']
        if func == Function.UDF:
            fn = kwargs.get('fn')
        else:
            dict_reduce_exception = {'mean': getattr(ff, 'reduce_mean'), 'sum': getattr(
                ff, 'sum_reduce'), 'count': getattr(ff, 'count')}
            fn = dict_reduce_exception[str(func)] if str(func) in dict_reduce_exception else getattr(ff, str(func))
            fn = fn(**kwargs)
        if fn is None:
            raise error
        try:
            output = fn(input_frame)
        except RuntimeError as e:
            print("Hit exception in reduce", e)
        return output
        # return qc.reduce(input_frame, fn)

    def run_node_reference(self, node_id, **kwargs):
        """Executes node reference"""
        print("Run node reference: ", node_id)
        offset = kwargs['_offset']
        if self.multi_input:
            return self.input[offset]
        assert offset == 0
        return self.input[0]

    def run_aux(self, node_id):
        """Executes auxiliary"""
        child_ids = self.plan.children(node_id)
        if child_ids:
            df = self.plan.result(child_ids[0])
            return df
        return self.input[0]

    def run_default_to_pandas(self, node_id, **kwargs):
        """Executes default to pandas"""
        child_ids = self.plan.children(node_id)
        input_frame = self.plan.result(child_ids[0])

        force_series = kwargs['force_series']
        df_method = kwargs['df_method']

        del kwargs['force_series']
        del kwargs['df_method']

        children_counter = 1
        args = []
        arg_i = 1
        current_kwarg_arg = "_arg_" + str(arg_i)
        while current_kwarg_arg in kwargs:
            arg_ = kwargs[current_kwarg_arg]
            if hasattr(arg_, "to_pandas"):
                arg_frame = self.plan.result(child_ids[children_counter])
                children_counter += 1

                args.append(arg_frame)
            else:
                args.append(arg_)
            del kwargs[current_kwarg_arg]

            arg_i += 1
            current_kwarg_arg = "_arg_" + str(arg_i)

        for key, val in kwargs.items():
            if val == "_Unprocessed_Lazy_DataFrame_":
                kwarg_frame = self.plan.result(child_ids[children_counter])
                children_counter += 1

                kwargs[key] = kwarg_frame
        return qc.default_to_pandas(input_frame, df_method, *args, force_series=force_series, **kwargs)

    def process_dest(self, result):
        """Processes destination by type and returns final result"""
        if isinstance(self.dest, dict):
            final_result = []
            if self.dest:
                for node_id in self.dest:
                    result = self.plan.result(node_id)
                    if self.dest[node_id]:
                        final_result.append(result)
                    else:
                        final_result = self.plan.result(self.root)
            else:
                final_result = self.plan.result(self.root)
        elif isinstance(self.dest, int):
            final_result = self.plan.result(self.dest)
        elif self.dest is None:
            final_result = self.plan.result(self.root)
        return final_result

    def select_func_call(self, name, func):
        """Selects function to call from the name"""
        if name == Operator.MAP1:
            func_call = self.run_map
        elif name == Operator.MAP2:
            func_call = self.run_map2
        elif name == Operator.REDUCEPARTITIONS:
            func_call = self.run_reduce
        elif name == Operator.SOURCE and func == Function.NODE:
            func_call = self.run_node_reference
        elif name == Operator.AUX:
            func_call = self.run_aux
        elif name == Operator.SINK:
            func_call = self.run_map
        elif name == Operator.GROUPBY and func == Function.GROUPBY:
            func_call = self.run_groupby
        elif name == Operator.DEFAULT_TO_PANDAS:
            func_call = self.run_default_to_pandas
        else:
            func_call = None
        return func_call

    def run(self, df_list):
        """Runs the plan with a postorder traversal and returns result"""
        # setup input and get the data from KV is needed
        self.input = df_list

        #return df_list[0]
        for root in self.plan.root_list:
            order_of_exec = self.plan.postorder_nodes(root)
            for node_id in order_of_exec:
                func = self.graph.nodes[node_id]['func']
                node_id = self.graph.nodes[node_id]['id']
                name = self.graph.nodes[node_id]['name']

                result = self.plan.result(node_id)

                if result is None:
                    # This node has not yet processed. Process it.
                    args = self.graph.nodes[node_id]['args']
                    # Preserve the original plan
                    # Make a copy of the plan arguments
                    # TODO: Do we need to preserve the internal rewr plan/opt plan?
                    ###kwargs = copy.deepcopy(args)
                    try:
                        func_call = self.select_func_call(name, func)
                        if func_call is None:
                            raise NotImplementedError("Function not yet supported in local_executor.")
                    except KeyError:
                        raise NotImplementedError("Function not yet supported.")
                    else:
                        try:
                            result = func_call(node_id, **args)
                            self.plan.cache(node_id, result)
                        except RuntimeError as e:
                            print("Exception, in local_exector", name, func, func_call, ":", e)

                if result is None:
                    raise RuntimeError("Result of an intermediate operator is None.")

        #TODO: change the coordinator/executor for batch mode
        return self.process_dest(result)
