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
"""MindPandas Lazy Mode SnapShot Class"""

import math
import time


class SnapShot:
    """
    This class is used to take snapshots of function calls and record execution times.
    """

    def __init__(self):
        self.func_ncalls = dict()  # A dict of func_name: number_of_calls
        self.func_call_time = dict()  # A dict of (func_name, nth_call): (entry_time, exit_time)

    def reset(self):
        """clear the `func_ncalls` and `func_call_time` dictionaries."""
        self.__init__()

    def dump(self):
        """Print a summary of the call times for each function."""
        result = dict()
        for func, ncalls in self.func_ncalls.items():
            result[func] = list()
            for nth in range(1, ncalls + 1):
                entry_t, exit_t = self.func_call_time[(func, nth)]
                elapse_t = exit_t - entry_t
                result[func].append((nth, elapse_t))
        for k, v in result.items():
            print(f"{k}: {v}")

    def dump_raw(self):
        """Print a raw dump of the `func_ncalls` and `func_call_time` dictionaries."""
        for func, ncalls in self.func_ncalls.items():
            print(f"{func}: {ncalls}")
            for nth in range(1, ncalls + 1):
                value = self.func_call_time[(func, nth)]
                print(f"{(func, nth)}: {value}")

    def dump_nan(self):
        """Print a summary of function calls that have `math.nan` as their exit time."""
        for func, ncalls in self.func_ncalls.items():
            num_nan = 0
            for nth in range(1, ncalls + 1):
                entry_t, exit_t = self.func_call_time[(func, nth)]
                if exit_t is math.nan:
                    if num_nan == 0:
                        print(f"{func}: {ncalls}")
                    num_nan += 1
                    if num_nan <= 3:
                        # Only print the first 3 entries
                        print(f"{(func, nth)}: {(entry_t, exit_t)}")
                    else:
                        print("...")
                        break

    def dump_acc_time(self):
        """Print the total elapsed time for each function."""
        result = dict()
        for func, ncalls in self.func_ncalls.items():
            result[func] = list()
            elapse_t = 0.0
            for nth in range(1, ncalls + 1):
                entry_t, exit_t = self.func_call_time[(func, nth)]
                elapse_t += exit_t - entry_t
            result[func].append(elapse_t)
        for k, v in result.items():
            if v[0] < pow(10, -2):
                print(f"time({k}) = {v[0] * 1000} ms")
            else:
                print(f"time({k}) = {v[0]} s")

    def dump_n_reset(self):
        """Print a summary of the call times for each function and reset the records."""
        self.dump()
        self.reset()

    def entry(self, name):
        """Record the entry time of a function call."""
        try:
            e = self.func_ncalls[name]
        except KeyError:
            self.func_ncalls[name] = 1
            self.func_call_time[(name, 1)] = (time.time(), math.nan)
        else:
            self.func_ncalls[name] += 1
            e += 1
            self.func_call_time[(name, e)] = (time.time(), math.nan)
        print("entry", name)

    def exit(self, name):
        """Record the exit time of a function call and updates the `func_ncalls` and `func_call_time` dictionaries."""
        try:
            e = self.func_ncalls[name]
        except KeyError:
            raise RuntimeError(f"No entry for function {name}.")
        else:
            entry_t, exit_t = self.func_call_time[(name, e)]
            assert exit_t != math.nan
            self.func_call_time[(name, e)] = (entry_t, time.time())
        print("exit", name)
