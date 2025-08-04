#!/bin/bash
# cloud_run.sh
#
# This script automates the process of building your Docker container,
# pushing it to Google Container Registry, and submitting a custom job
# to Vertex AI for either data generation or training.

set -e # Exit immediately if a command exits with a non-zero status.

# --- CONFIGURATION ---
# --- BITTE DIESE WERTE ANPASSEN ---

# 1. Google Cloud Project-ID
export PROJECT_ID="osudaily" # z.B. "my-awesome-project-12345"

# 2. GCS-Bucket für Sourcen, Daten und Ergebnisse
export GCS_BUCKET="gs://osudaily_dataset" # z.B. "gs://my-osu-data-bucket"

# 3. Region für den Vertex AI Job
export REGION="europe-west3" # z.B. "us-central1", "europe-west4"

# 4. Name für das Docker-Image
export IMAGE_NAME="osu-dataset-generator"

# 5. Eindeutiger Name für den Vertex AI Job
export JOB_NAME="osu_job_$(date +%Y%m%d_%H%M%S)"

# --- ENDE DER KONFIGURATION ---

# --- DOCKER IMAGE URI ---
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

# --- FUNKTIONEN ---

build_and_push_docker() {
    echo "--- Schritt 1: Docker-Image bauen und hochladen ---"
    echo "Image URI: ${IMAGE_URI}"

    # Konfiguriere Docker für die Authentifizierung mit gcloud
    gcloud auth configure-docker -q

    # Baue das Docker-Image
    docker build -t "${IMAGE_URI}" .

    # Lade das Image in die Google Container Registry hoch
    docker push "${IMAGE_URI}"

    echo "--- Docker-Image erfolgreich hochgeladen ---"
}

run_data_generation_job() {
    echo "--- Schritt 2: Vertex AI Job zur Datengenerierung starten ---"
    echo "Job-Name: ${JOB_NAME}_datagen"

    # Dies ist ein leistungsstarker CPU-Job für die parallele Bilderzeugung.
    # Sie können den Maschinentyp an Ihre Bedürfnisse anpassen.
    # e.g., n1-standard-16 (16 vCPUs), n1-highcpu-32 (32 vCPUs)
    # Siehe: https://cloud.google.com/vertex-ai/docs/training/configure-compute
    gcloud ai custom-jobs create \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --display-name="${JOB_NAME}_datagen" \
      --worker-pool-spec="machine-type=n1-highcpu-16,replica-count=1,container-image-uri=${IMAGE_URI}" \
      --args="python,master_pipeline.py,--base-gcs-bucket=${GCS_BUCKET}"

    echo "--- Datengenerierungs-Job gestartet. Überwachen Sie den Fortschritt in der Google Cloud Console. ---"
}

run_training_job() {
    echo "--- Schritt 2: Vertex AI Job zum Modelltraining starten ---"
    echo "Job-Name: ${JOB_NAME}_training"

    # Der Pfad zur YAML-Datei, die vom Datengenerierungs-Skript erstellt wurde.
    # Passen Sie FINAL_DATASET_NAME an, wenn Sie ihn in master_pipeline.py geändert haben.
    local final_dataset_name="master_dataset_v16"
    local yaml_path="${GCS_BUCKET}/datasets/${final_dataset_name}/${final_dataset_name}.yaml"
    local output_path="${GCS_BUCKET}/datasets/${final_dataset_name}/runs"

    # Dies ist ein GPU-Job.
    # machine-type: n1-standard-8 (8 vCPUs, 30GB RAM)
    # accelerator: type=nvidia-tesla-t4,count=1 (Eine T4 GPU)
    # Andere Optionen: nvidia-l4, nvidia-tesla-a100
    gcloud ai custom-jobs create \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --display-name="${JOB_NAME}_training" \
      --worker-pool-spec="machine-type=n1-standard-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=${IMAGE_URI}" \
      --args="python,train.py,--data=${yaml_path},--gcs-output-path=${output_path},--batch-size=16" # Passen Sie die Batch-Größe an Ihre GPU an

    echo "--- Trainings-Job gestartet. Überwachen Sie den Fortschritt in der Google Cloud Console. ---"
}


# --- SKRIPT-AUSFÜHRUNG ---

# Überprüfen, ob ein Argument übergeben wurde
if [ "$#" -ne 1 ]; then
    echo "Fehler: Bitte geben Sie einen Modus an."
    echo "Verwendung: ./cloud_run.sh [build|datagen|train]"
    exit 1
fi

MODE=$1

gcloud config set project "${PROJECT_ID}"

if [ "$MODE" == "build" ]; then
    build_and_push_docker
elif [ "$MODE" == "datagen" ]; then
    run_data_generation_job
elif [ "$MODE" == "train" ]; then
    run_training_job
else
    echo "Fehler: Ungültiger Modus '$MODE'."
    echo "Verfügbare Modi: build, datagen, train"
    exit 1
fi

echo "--- Skript beendet. ---"

# NEU: Hält das Fenster offen, damit Sie Fehler lesen können.
read -p "Druecken Sie die [Enter]-Taste, um das Fenster zu schliessen..."

