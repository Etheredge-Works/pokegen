model=$1
version=$2

echo "# $model ($version)"

echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target reports/$model/$version/logs.json --show-md

echo "## Generated"
gen_dir=reports/$model/$version/gen
for f in $(ls $gen_dir/*.gif)
do
    echo "Publishing $f"
    cml-publish "$f" --md 
done