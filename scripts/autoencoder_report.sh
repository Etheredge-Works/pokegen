#! /bin/bash 
model=$1
version=$2
log_dir=reports/$model/$version

echo "<details>"
echo "    <summary>$model ($version)</summary>"
echo ""

echo "# $model ($version)"

echo ""
echo "\`\`\`"
cat $log_dir/summary.txt
echo "\`\`\`"

echo "## Metrics"
git fetch --prune >& /dev/null
dvc metrics diff main --target $log_dir/logs.json --show-md

echo "### Loss"
#dvc plots show --show-vega $log_dir/logs/loss.tsv $log_dir/logs/val_loss.tsv > /tmp/vega.json
#vl2png /tmp/vega.json | cml-publish --md

dvc plots diff --target $log_dir/logs/loss.tsv --show-vega main > /tmp/vega.json
vl2png /tmp/vega.json | cml-publish --md

dvc plots diff --target $log_dir/logs/val_loss.tsv --show-vega main > /tmp/vega.json
vl2png /tmp/vega.json | cml-publish --md


echo "### LR"
dvc plots diff --target $log_dir/logs/lr.tsv --show-vega main > /tmp/vega.json
vl2png /tmp/vega.json | cml-publish --md


echo "## Latent Space"
latent_dir=$log_dir/latent

for f in $(ls $latent_dir/*/*.png)
do
    echo $f
    cml-publish "$f" --md 
done


echo "## Encoded/Decoded"
results_dir=reports/$model/$version/results
for f in $(ls $results_dir)
do
    echo $f
    cml-publish "$results_dir/$f/raw.jpg" --md 
    cml-publish "$results_dir/$f/encdec.jpg" --md 
    echo "---"
done


echo "## Generated"
gen_dir=$log_dir/gen
for f in $(ls $gen_dir/*gif)
do
    cml-publish "$f" --md 
done

echo "</details>"