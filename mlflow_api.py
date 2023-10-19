import mlflow

mlflow.tracking.set_registry_uri("http://127.0.0.1:5000/")

print(mlflow.tracking.get_tracking_uri())

experiment = mlflow.get_experiment("415665051034090980")
print("Name: {}".format(experiment.name))

experiment_id = mlflow.get_experiment_by_name("415665051034090980")
print("ID: {}".format(experiment_id))

with mlflow.start_run(run_name="new-run1-10") as run5:
    last_run = mlflow.last_active_run().info.run_id
    print("last_run ", last_run)