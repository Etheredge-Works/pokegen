#! /bin/bash
./scripts/autoencoder_report.sh autoencoder dense > report.md
./scripts/autoencoder_report.sh autoencoder conv >> report.md
./scripts/autoencoder_report.sh vae dense >> report.md
./scripts/autoencoder_report.sh vae conv >> report.md