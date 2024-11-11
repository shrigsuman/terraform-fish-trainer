import clearml
from clearml import PipelineDecorator, Task, Dataset
import os
from ultralytics import YOLO


Task.set_credentials(
     api_host="https://api.clear.ml",
     web_host="https://app.clear.ml",
     files_host="https://files.clear.ml",
     key='4O1TKN45P9T23DU5TW3RPU6WWW4LY5',
     secret='OgQwh3puKCIq3CqfZSCZUlMIhOnVU1m8wNAQ7szcVa8MA3fHHmaDpHhEhbKvYK7R590'
)

# ============ Login to clearML ===========================
task = Task.init(project_name="flytechy-ai-ml", task_name="testing-fish-pose-estimation-v6-k14")
dataset_id_with_visibility_all_data = "415fa495853e48799539841f0360623b"

# ============= download dataset and setup config ==========================
try:
    dataset = Dataset.get(dataset_id=dataset_id_with_visibility_all_data)
    dataset_path = dataset.get_mutable_local_copy("training_dataset")
    kpt_dim = 2

    config_data = """
    path: {dataset_path}
    train: images/train  # train images (relative to 'path')
    val: images/val  # val images (relative to 'path')

    # Keypoints
    kpt_shape: [14, {kpt_dim}]  # [number of keypoints, number of dim]
    names:
    0: fish
    """.format(
        dataset_path=dataset_path,
        kpt_dim=kpt_dim
        )

    with open("config.yaml", 'w') as file:
        file.write(config_data)

except Exception as e:
    print(f"Exception: {str(e)}")


# ============== download pretrained model from s3 bucket ==============
# boto3



# ================== model training ====================================
model = YOLO("yolo11n-pose.pt") # loading pretrained model

num_epochs = 2
model.train(data='config.yaml',
            epochs=num_epochs,
            imgsz=640,
            augment=True,
            close_mosaic=400,
            single_cls=True,
            pose=13.5,
            lr0=0.09,
            lrf=0.006,
            patience=100
            )

#========================= close task ===========================
task.close()

# ================= zip runs ====================================
# ============= upload zipped runs into s3 bucket ===============


# <<======== terminate ec2 instance =======>>> [no need for now]
