model=$1
version=$2

echo "# $model ($version)"

echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target reports/$model/$version/logs.json --show-md

echo "## Encoded/Decoded"
gen_dir=reports/$model/$version/results
do
    cml-publish "$f/raw.jpg" --md 
    cml-publish "$f/result.jpg" --md 
    echo "---"
done

echo "## Generated"
gen_dir=reports/$model/$version/gen
do
    cml-publish "$f" --md 
done