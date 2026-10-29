# Brain MRI Classification and Detection

An end-to-end brain MRI project for image-level tumor classification and
bounding-box detection. It trains TensorFlow/Keras classifiers, trains an
Ultralytics YOLO detector, and provides a Streamlit app for interactive
inference and evaluation-report viewing. Docker Compose can run the app and an
optional Airflow training pipeline.

> **Medical disclaimer:** This project is for research and demonstration only.
> Its predictions are not a medical diagnosis and must not be used as a
> substitute for advice from a qualified medical professional.

## Contents

- [Models and classes](#models-and-classes)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Get the data](#get-the-data)
- [Prepare and validate the data](#prepare-and-validate-the-data)
- [Train the models](#train-the-models)
- [Evaluate the detector](#evaluate-the-detector)
- [Run the Streamlit app](#run-the-streamlit-app)
- [Run with Docker Compose](#run-with-docker-compose)
- [Run the Airflow pipeline](#run-the-airflow-pipeline)
- [Outputs and reports](#outputs-and-reports)
- [Troubleshooting](#troubleshooting)

## Models and classes

The classification models predict one of four image-level classes:

| Class ID | Classification label |
|---:|---|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

Classification is performed by either **MobileNetV2** or **ResNet50**. The
inference app resizes uploaded images to 224 × 224 pixels and applies the
preprocessing function for the selected model.

The YOLO detector predicts tumor bounding boxes for **Glioma**, **Meningioma**,
and **Pituitary**. **No Tumor** images are used as background examples rather
than as a detection class.

The detector uses YOLOv8n and the Ultralytics YOLO format. Each YOLO annotation
row is a class ID followed by normalized center-x, center-y, width, and height:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image dimensions.

## Repository layout

```text
app/                    Streamlit home page, inference pages, and shared helpers
airflow/dags/            Airflow DAG for the data and model pipeline
configs/                 Project configuration placeholders
data/
  raw/                   Source dataset (not tracked by Git)
  processed/             Corrected data and generated splits (not tracked)
  samples/               Optional local sample images
docker/                  Airflow Dockerfile
models/                  Trained model weights (not tracked by Git)
reports/                 Training, dataset-quality, and evaluation reports
src/
  data/                  Dataset download, validation, and preparation
  training/              Classifier and detector training
  evaluation/            Detector evaluation
Dockerfile               Streamlit image
docker-compose.yml       Streamlit and optional Airflow services
requirements.txt         Python dependencies for the app and ML code
```

The `configs/*.yaml` files are currently empty placeholders. The scripts use
their Python-defined settings and generate the YOLO dataset configuration under
`data/processed/mri_detection/yolo_data.yaml`.

## Requirements

For local development:

- Python 3.11 is recommended to match the supplied Docker images.
- Git.
- Enough disk space for the MRI dataset, generated datasets, model weights,
  and training reports. ResNet50 weights and training artifacts can be large.
- Optional: a CUDA-compatible GPU and compatible TensorFlow/Ultralytics
  installations for accelerated local training.

For Docker:

- Docker Desktop on Windows or Docker Engine on Linux/macOS.
- Docker Compose v2.

## Get the data

The repository does not include the dataset. The helper
[`data/download_data_set.py`](./data/download_data_set.py) downloads the
Kaggle-hosted dataset `ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes`
using KaggleHub:

```powershell
python -m pip install kagglehub
python data\download_data_set.py
```

The helper prints the KaggleHub cache location; it does not place the dataset
in this repository automatically. Copy or arrange the downloaded data under
`data/raw/` in the folder structure expected by the preparation scripts.
Follow Kaggle's current authentication requirements if the dataset download
requires an account or credentials.

Expected input folders:

```text
data/raw/
├── Train/
│   ├── Glioma/
│   │   ├── images/
│   │   └── labels/
│   ├── Meningioma/
│   │   ├── images/
│   │   └── labels/
│   ├── No Tumor/
│   │   ├── images/
│   │   └── labels/
│   └── Pituitary/
│       ├── images/
│       └── labels/
└── Val/
    └── <same class folders and images/labels structure>
```

Images and annotation files are matched by filename stem. The class folder
names and class IDs must use the mapping documented above. Images without
labels are allowed for `No Tumor`; tumor images need valid YOLO annotation
files.

Check the source dataset's terms and licensing before redistributing it or
trained artifacts derived from it.

## Prepare and validate the data

Run commands from the repository root.

1. Inspect the raw data and annotation examples:

   ```powershell
   python -m src.data.load_data
   ```

2. Check image/annotation consistency and normalized bounding-box values:

   ```powershell
   python -m src.data.validate_data
   ```

3. Preview proposed class-folder corrections. This step does not modify the
   source dataset unless `--apply` is passed:

   ```powershell
   python -m src.data.correct_class_folders
   ```

   Apply corrections to a copied, corrected dataset:

   ```powershell
   python -m src.data.correct_class_folders --apply
   ```

   The command uses annotation class IDs to correct mislabeled folders and
   quarantines images with ambiguous multi-class labels. Review its summary
   before continuing.

4. Create classification manifests and deterministic train, validation, and
   test splits:

   ```powershell
   python -m src.data.prepare_dataset
   ```

   This reads from `data/processed/mri_classification_corrected/`. It
   de-duplicates images by pixel content, reserves the source `Val` split for
   testing, and divides the source `Train` split into training and validation
   data. Generated CSV manifests and a split summary are written under
   `data/processed/mri_dataset/`.

5. Prepare the YOLO detection dataset and its quality report:

   ```powershell
   python -m src.data.create_yolo_config
   ```

   This creates `data/processed/mri_detection/`, generates
   `yolo_data.yaml`, and writes a dataset quality summary under
   `reports/detection/`. Detection data is generated from the classification
   splits. The converter maps source class IDs 0, 1, and 3 to the three tumor
   detection classes; No Tumor images are treated as background. Review the
   quality summary and rejected-image CSV before detector training.

The classification training loader reads
`data/processed/mri_classification_corrected/Train` and `Val`. Ensure the
corrected folder tree contains each class's `images/` and `labels/` folders,
including an empty `labels/` folder for No Tumor if there are no label files.

## Install local dependencies

Create and activate a virtual environment in PowerShell, then install the
project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, either adjust the execution
policy according to your organization's guidance or run the environment's
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file supports the Streamlit app and ML scripts. Airflow is
installed separately by the Airflow Docker image. KaggleHub is only needed if
you use the optional dataset download helper.

## Train the models

Prepare the data first. Train either classifier from the repository root:

```powershell
python -m src.training.train_classifier --model mobilenetv2
python -m src.training.train_classifier --model resnet50
```

Each classifier trains a frozen feature extractor and then fine-tunes selected
backbone layers. It selects the better checkpoint based on validation loss,
evaluates the selected model on the test split, and writes the final Keras
model to:

```text
models/mobilenetv2/best_model.keras
models/resnet50/best_model.keras
```

Train the YOLO detector after generating the YOLO dataset:

```powershell
python -m src.training.train_detector
```

Detector training uses `yolov8n.pt` as its pretrained starting point. Ultralytics
downloads that checkpoint if it is not already available. On a CUDA machine,
the training script defaults to device `0`; override it to use CPU or another
supported device:

```powershell
$env:YOLO_DEVICE = "cpu"
python -m src.training.train_detector
```

The trained detector is copied to `models/detection/best.pt`. Training runs and
checkpoints are also saved under `reports/detection/yolov8n_brain_mri/`.

## Evaluate the detector

After detector training, evaluate against the test split:

```powershell
python -m src.evaluation.evaluate_detector
```

The evaluation writes `reports/detection/test_metrics.json` and Ultralytics
plots under `reports/detection/yolov8n_test_evaluation/`. It reports precision,
recall, mAP@50, mAP@75, mAP@50–95, and per-class mAP@50–95.

Classifier evaluation runs as part of classifier training. The resulting
metrics include test loss and accuracy.

## Run the Streamlit app

### Local Python

From the project root, after installing dependencies and placing model weights
at the expected paths:

```powershell
python -m streamlit run app\streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

The app has three pages:

- **Classification:** upload an image and select MobileNetV2 or ResNet50 to
  view the predicted class and class probabilities.
- **Detection:** upload an image, select a minimum confidence threshold, and
  view YOLO detections and bounding boxes.
- **Model Metrics:** view available classifier reports and detector test
  metrics.

Supported upload formats are JPG, JPEG, PNG, BMP, TIF, TIFF, and WebP.

Expected weights:

| Model | Expected path |
|---|---|
| MobileNetV2 | `models/mobilenet/best_model.keras` or `models/mobilenetv2/best_model.keras` |
| ResNet50 | `models/resnet50/best_model.keras` |
| YOLO detector | `models/detection/best.pt` |

The app loads model weights on demand and caches loaded models for the current
Streamlit process.

## Run with Docker Compose

Docker keeps model weights, reports, and datasets outside the images. Compose
mounts the local `models/` and `reports/` folders read-only in the app
container, and mounts `data/`, `models/`, and `reports/` into Airflow so
training data and outputs persist on the host.

Build and start only the Streamlit app:

```powershell
docker compose up --build app
```

Open [http://localhost:8501](http://localhost:8501).

Start the app in the background:

```powershell
docker compose up --build -d app
docker compose logs -f app
```

Stop the app:

```powershell
docker compose down
```

The app image runs as a non-root user and has a Streamlit health check. Models
must be present in the host `models/` directory before inference.

## Run the Airflow pipeline

Airflow is included as an opt-in Compose profile. Start it with:

```powershell
docker compose --profile airflow up --build airflow
```

Open [http://localhost:8080](http://localhost:8080). Airflow standalone
initializes a local development instance; its startup logs include the
generated login credentials:

```powershell
docker compose logs airflow
```

In the Airflow UI, enable and trigger the `brain_mri_pipeline` DAG. The DAG
performs folder correction, dataset preparation, MobileNetV2 training, ResNet50
training, YOLO dataset preparation, detector training, and detector evaluation.
The Airflow service uses CPU for YOLO by default.

To select a device for YOLO training, set `YOLO_DEVICE` before starting
Compose, for example:

```powershell
$env:YOLO_DEVICE = "cpu"
docker compose --profile airflow up --build airflow
```

To run both services:

```powershell
docker compose --profile airflow up --build
```

Airflow runs with a local SQLite database using the standalone command. This
setup is intended for local development and experimentation, not production.
A production deployment should use an Airflow-supported metadata database,
appropriate authentication, secrets management, and a production executor.

## Outputs and reports

| Output | Location |
|---|---|
| Corrected classification dataset | `data/processed/mri_classification_corrected/` |
| Classification split manifests | `data/processed/mri_dataset/` |
| Generated YOLO dataset and config | `data/processed/mri_detection/` |
| Classifier model weights | `models/mobilenetv2/` and `models/resnet50/` |
| Detector model weight used by the app | `models/detection/best.pt` |
| Detector training run | `reports/detection/yolov8n_brain_mri/` |
| Detector test metrics and plots | `reports/detection/` |
| Classifier training metrics | `reports/mobilenetv2/` and `reports/resnet50/` |

The metrics page discovers classifier evaluation reports under
`reports/config*/<model>/metrics.json` and a detector report at
`reports/detection/test_metrics.json`. The classifier training script writes
its own metrics to `reports/<model>/metrics.json`; copy or organize reports
into the per-configuration layout if you want those runs displayed by the
metrics page.

## Troubleshooting

### The app says a model is missing

Check that the model file exists at one of the expected paths in the
[Streamlit app section](#run-the-streamlit-app). Model weights are not included
in the Git repository or Docker image.

### The detector dataset has rejected images or missing classes

Review `reports/detection/dataset_quality.json` and
`data/processed/mri_detection/rejected_images.csv`. Confirm that image and
annotation stems match, class IDs follow the documented mapping, and tumor
training images contain valid boxes for all three detection classes.

### KaggleHub cannot download the dataset

Install KaggleHub, check the dataset's current access requirements, and follow
Kaggle's authentication instructions. The download helper prints a cache
location; move or copy the downloaded files into the expected `data/raw/`
structure before running preprocessing.

### Airflow cannot find the DAG or its data

Start Compose from the repository root with the `airflow` profile. Confirm the
source dataset is mounted under `data/` and review service logs:

```powershell
docker compose --profile airflow logs airflow
```

### Docker is unavailable

Make sure Docker Desktop or the Docker Engine is running, then check Compose:

```powershell
docker version
docker compose version
```

## Data, artifacts, and responsible use

Dataset files, model weights, and generated reports are excluded from Git by
the repository's ignore rules. Store backups separately and verify that you
have permission to use and share source data and derived artifacts.

This software is an educational research project, not a medical device. Model
performance on the available evaluation set does not establish clinical
validity, safety, or generalization to other scanners, patient populations, or
imaging protocols. Do not use the models to make clinical decisions.