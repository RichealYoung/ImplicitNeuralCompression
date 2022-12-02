from datetime import datetime
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
from omegaconf import OmegaConf
import argparse
from utils.multitask_config_process import omegaconf2dotlist, CONCAT, omegaconf2dict
from utils.tasksmanager import Task, Queue

timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]


def generate_task_list(config_path: str, main_script_path: str):
    temp_output_dir = opj(opd(__file__), "outputs", "BatchExp" + timestamp)
    os.makedirs(temp_output_dir)
    # read BatchExp config
    config = OmegaConf.load(config_path)
    # parse each task config
    static = omegaconf2dotlist(config.Static)
    dynamic_list = CONCAT(config.Dynamic)
    dotlist_list = [static + dynamic for dynamic in dynamic_list]
    # generate task
    task_list = []
    for task_idx, dotlist in enumerate(dotlist_list):
        task_config = OmegaConf.from_dotlist(dotlist)
        task_config_save_path = opj(temp_output_dir, str(task_idx) + ".yaml")
        OmegaConf.save(task_config, task_config_save_path)
        # devide = False if task_opt_yaml.devide.type == "None" else True
        command = "python {} -c {}".format(main_script_path, task_config_save_path)
        stdout_save_path = opj(temp_output_dir, str(task_idx) + ".log")
        # task_list.append(Task(command, stdout, devide))
        task_list.append(Task(command, stdout_save_path))
    return task_list, temp_output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Compression")
    parser.add_argument(
        "-c",
        type=str,
        default=opj(opd(__file__), "config", "BatchExp", "sci.yaml"),
        help="config file path",
    )
    parser.add_argument(
        "-stp",
        type=str,
        default=opj(opd(__file__), "sci.py"),
        help="the singletask script path",
    )
    parser.add_argument(
        "-g",
        help="availabel gpu list",
        default="0,1,2,3",
        type=lambda s: [int(item) for item in s.split(",")],
    )
    parser.add_argument(
        "-t",
        type=float,
        default=2,
        help="the time interval between each task-assigning loop",
    )
    args = parser.parse_args()
    task_list, temp_output_dir = generate_task_list(args.c, args.stp)
    queue = Queue(task_list, args.g)
    queue.start(args.t, remind=True)
    # try:
    #     queue = Queue(task_list, args.g)
    #     queue.start(args.t, remind=True)
    #     shutil.rmtree(temp_output_dir)
    # except:
    #     shutil.rmtree(temp_output_dir)
    #     pass
