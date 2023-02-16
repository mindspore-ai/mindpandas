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
Module contains ``Client`` class which provides a coordinator to run the plan
"""
import mindpandas as mpd
import mindpandas.dist.coordinator as co


class Client:
    """
    Client provides coordinator for running the lazy plan.
    """
    def __init__(self, plan, debug, pr_stats, pr_details):
        self.plan = plan
        self.debug = debug
        self.pr_stats = pr_stats
        self.pr_details = pr_details

    def run(self):
        concurrency = mpd.get_concurrency_mode()
        coordinator = co.CoordinatorActor(self.plan, self.debug, self.pr_stats, self.pr_details,
                                          cfg_concurrency=concurrency)
        result = coordinator.run()
        return result
