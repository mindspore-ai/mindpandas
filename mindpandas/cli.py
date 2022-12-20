#!/usr/bin/env python3
# Copyright 2022 Huawei Technologies Co., Ltd
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

""" command line tool """
import os
import subprocess
import multiprocessing

import click
import psutil


def _get_python_bin_path() -> str:
    site_packages_root = os.path.dirname(_get_install_root())
    return os.path.join(site_packages_root, "../../bin/python3.8")


def _get_install_root() -> str:
    """
    get fleeting path in site-packages
    :return fleeting path
    """
    current_path = os.path.abspath(__file__)
    return os.path.dirname(current_path)


def _get_yrctl_bin_path() -> str:
    """
    get yrctl bin path
    :return:
    """
    yrctl_path = os.path.join(_get_install_root(), "dist_executor/modules/bin/mpctl")
    if not os.path.isfile(yrctl_path):
        click.echo(f"failed to find {yrctl_path}")
        return ""
    return yrctl_path


def _execute_cmd(cmd: str) -> str:
    process = subprocess.Popen(
        cmd.split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    for line in iter(process.stdout.readline, b""):
        print(line.decode("utf-8"), end="")
    process.stdout.close()
    process.wait()
    return ""


@click.group(help="The distributed executor of MindPandas.")
def yrctl():
    pass


@click.command(help="used to start the fleeting cluster")
@click.option("--master", is_flag=True, help='set if starting a master node')
@click.option("-a", "--address", type=str, default=None, help='the ip address of the master node')
@click.option("--cpu", type=int, default=None, help='number of cpus to use')
@click.option("--datamem", type=int, default=None, help='amount of memory used by datasystem')
@click.option("--mem", type=int, default=None, help='amount of general purpose memory')
@click.option("--tmp-dir", type=str, default='/tmp/mindpandas/', help='temporary directory for the mindpandas process')
@click.option("--tmp-file-size-limit", type=int, default=None, help='size limit of the temporary files (MB)')
def start(master, address, cpu, datamem, mem, tmp_dir, tmp_file_size_limit):
    """The command to start distributed executor service."""
    if not os.path.isabs(tmp_dir):
        raise ValueError(f'"{tmp_dir}" is not a valid path')
    if cpu is None:
        cpu = multiprocessing.cpu_count()
    system_memory = psutil.virtual_memory().total // (1 << 20)  # Available memory in MB.
    if datamem is None and mem is None:
        datamem = int(system_memory * 0.3)
        mem = int(system_memory * 0.9)
    local_address = address if master else None
    options = ['-m' if master else '',
               f'-a {address}' if address is not None else '',
               f'--localaddress {local_address}' if local_address is not None else '',
               f'-p 4b7cffe5c9ca38db3db3bbabdf858c10',
               f'--cpu {cpu * 1000}' if cpu is not None else '',
               f'--datamem {datamem}' if datamem is not None else '',
               f'--mem {mem}' if mem is not None else '',
               f'--spillPath {os.path.join(tmp_dir, "mp-")}' if tmp_dir is not None else '',
               f'--spillLimit {tmp_file_size_limit}' if tmp_file_size_limit is not None else '']

    option_str = ' '.join(options)
    print(f"Starting distributed executor with option:\n"
          f"address={address}, "
          f"cpu={cpu}, "
          f"datamem={datamem}, "
          f"mem={mem}, "
          f"spillPath={tmp_dir}, "
          f"spillLimit={tmp_file_size_limit}\n")
    cmd = f"{_get_yrctl_bin_path()} start {option_str}"
    subprocess.run(cmd, check=True, shell=True)


@click.command(help="used to stop the fleeting cluster")
def stop():
    os.system(f"{_get_yrctl_bin_path()} stop")


yrctl.add_command(start)
yrctl.add_command(stop)

if __name__ == '__main__':
    yrctl()
