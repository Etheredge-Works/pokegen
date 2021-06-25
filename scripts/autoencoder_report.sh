model=$1
version=$2

echo "# $model ($version)"

echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target reports/$model/$version/logs.json --show-md

echo "## Encoded/Decoded"
results_dir=reports/$model/$version/results
for f in $(ls $results_dir)
do
    echo $f
    cml-publish "$results_dir/$f/raw.jpg" --md 
    cml-publish "$results_dir/$f/result.jpg" --md 
    echo "---"
done

echo "## Generated"
gen_dir=reports/$model/$version/gen
for f in $(ls $gen_dir/*gif)
do
    cml-publish "$f" --md 
done
