echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target reports/autoencoder/logs.json
