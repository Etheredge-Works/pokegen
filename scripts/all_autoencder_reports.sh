#! /bin/bash
echo "# Report" 
dvc params diff main --show-md
./scripts/autoencoder_report.sh ae dense
./scripts/autoencoder_report.sh ae conv 
./scripts/autoencoder_report.sh vae dense
./scripts/autoencoder_report.sh vae conv