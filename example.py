import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Get the list of all tasks
# task_ids = r.lrange('taskiq_redis:tasks', 0, -1)
task_ids = r.lrange('taskiq:queue', 0, -1)
print(task_ids)
# Check the status of each task
all_tasks_completed = True
for task_id in task_ids:
    status = r.hget(f'taskiq_redis:task:{task_id}', 'status')
    if status != b'completed':
        all_tasks_completed = False
        break

if all_tasks_completed:
    print("All tasks are completed.")
else:
    print("Not all tasks are completed.")
