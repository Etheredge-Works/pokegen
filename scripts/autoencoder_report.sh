version=$1
echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target reports/autoencoder/$version/logs.json --show-md
