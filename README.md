# Brain MRI classification and detection

The project provides MobileNetV2 and ResNet50 image classifiers, a YOLO tumor
detector, and a Streamlit inference app. Docker Compose can run the app and the
Airflow training pipeline in separate containers.

## Requirements

- Docker Desktop or Docker Engine with Docker Compose
- Trained model weights in `models/` to run the app
- The source datasets in `data/` to run the Airflow training pipeline

Docker Compose keeps model weights, reports, and datasets in the project
directories on the host. These files are not included in the Docker images.

## Run the Streamlit app

From the project root:

```powershell
docker compose up --build app
```

Open [http://localhost:8501](http://localhost:8501).

The app expects classifier weights at
`models/mobilenet/best_model.keras` (or
`models/mobilenetv2/best_model.keras`) and
`models/resnet50/best_model.keras`, plus detector weights at
`models/detection/best.pt`.

## Run the Airflow pipeline

Start Airflow with its optional Compose profile:

```powershell
docker compose --profile airflow up --build airflow
```

Open [http://localhost:8080](http://localhost:8080). The Airflow standalone
command prints the initial login credentials in its logs:

```powershell
docker compose logs airflow
```

Enable the `brain_mri_pipeline` DAG in the Airflow UI and trigger it when the
datasets are available under `data/`. The pipeline trains both classifiers and
the detector, then evaluates the detector. Training outputs persist under
`models/` and `reports/`. YOLO uses CPU by default in the Airflow container;
Set `YOLO_DEVICE` to a device supported by your Docker GPU runtime and
configure the matching GPU device request in Compose to use a GPU.

To run both the app and Airflow together:

```powershell
docker compose --profile airflow up --build
```

Airflow standalone with its local SQLite database is intended for local
development, not production deployment. The model outputs are for research and
demonstration only and are not a medical diagnosis.