from datetime import datetime
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = Path(__file__).resolve().parents[2]

with DAG(
    dag_id="brain_mri_pipeline",
    description="MRI classification and tumor detection pipeline",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 0,
    },
    tags=["brain-mri", "classification", "detection"],
) as dag:

    correct_class_folders = BashOperator(task_id="correct_class_folders", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.data.correct_class_folders --apply"))

    prepare_dataset = BashOperator(task_id="prepare_dataset", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.data.prepare_dataset"))

    train_classifier_mobilenet = BashOperator(task_id="train_classifier_mobilenet", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.training.train_classifier --model mobilenetv2"))

    train_classifier_resnet = BashOperator(task_id="train_classifier_resnet", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.training.train_classifier --model resnet50"))

    create_yolo_config = BashOperator(task_id="create_yolo_config", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.data.create_yolo_config"))

    train_detector = BashOperator(task_id="train_detector", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.training.train_detector"))

    evaluate_detector = BashOperator(task_id="evaluate_detector", bash_command=(f"cd {PROJECT_DIR} && " "python -m src.evaluation.evaluate_detector"))

    (
        correct_class_folders
        >> prepare_dataset
        >> train_classifier_mobilenet
        >> train_classifier_resnet
        >> create_yolo_config
        >> train_detector
        >> evaluate_detector
    )
