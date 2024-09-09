import re
import os
import fcntl
import time
from pathlib import Path


def sanitize_filename(filename):
    forbidden_chars = r'[\/\\\?\%\*\:\|\"<>\.]'
    
    sanitized = re.sub(forbidden_chars, '', filename)
    
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    max_length = 255
    sanitized = sanitized[:max_length]
    
    return sanitized

def read_and_lock_file(file_path):
    with open(file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        content = f.readlines()
        return content

def write_and_unlock_file(file_path, content):
    with open(file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        f.writelines(content)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)

def initialize_task_file(queue_name, start_idx, end_idx):
    lock_file = "lock.lock"
    with open(lock_file, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        
        task_file = f"{queue_name}_task.txt"
        if not Path(task_file).exists():
            with open(task_file, 'w') as f:
                for i in range(start_idx, end_idx):
                    f.write(f"{i} pending\n")
        
        fcntl.flock(lock_f, fcntl.LOCK_UN)

def get_next_task(queue_name):
    task_file = f"{queue_name}_task.txt"
    tasks = read_and_lock_file(task_file)
    next_task = None
    
    for i in range(len(tasks)):
        try:
            task_id, status = tasks[i].strip().split()
            task_id = int(task_id)
        except ValueError:
            print(f"Invalid task line: {tasks[i]}")
            continue
        
        if status == 'pending':
            next_task = task_id
            tasks[i] = f"{task_id} in_progress\n"
            break
    
    write_and_unlock_file(task_file, tasks)
    return next_task

def mark_task_done(queue_name, task_id):
    task_file = f"{queue_name}_task.txt"
    tasks = read_and_lock_file(task_file)
    for i in range(len(tasks)):
        task_line = tasks[i].strip()
        if task_line.startswith(f"{task_id}"):
            tasks[i] = f"{task_id} done\n"
            break
    write_and_unlock_file(task_file, tasks)