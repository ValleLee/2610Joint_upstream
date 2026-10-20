# Data lists and external files

`pairs/k24_ratio100_val2_ignore/{train,val}_pairs.txt` are the exact five-column
lists from the 2026-10-02 JointTeacher run. They contain 36,430 training and 715
validation rows, including `_ref` rows. The columns are rendered RGB, rendered
depth NPY, DA(GT RGB) NPY, DA(rendered RGB) NPY, and GT RGB. The training dataset
uses columns 1, 2, 3, and 5; column 4 is retained for the post-training DA
comparison.

The lists preserve historical absolute paths on the original machine. The image
and depth datasets themselves are external and are not included in this package.
For another machine, use `script/remap_pair_paths.py` to write local copies, then
set `PAIRS_DIR` to the generated directory before starting training. Example:

```bash
python3 script/remap_pair_paths.py \
  --replace /data2/caoxiaodata/data=/new/datasets \
  --replace /data1/lilinwei/Depth-Prior-Optimization-for-SVGS/data=/new/da_data \
  --output-dir data/pairs/local --check-files
PAIRS_DIR="$PWD/data/pairs/local" bash script/train_rgbdepth_k24_r100_perimg.sh
```

Alternatively, create the lists from an external four-column source using
`python3 script/build_joint_pairs.py --source-dir ... --gt-root ...`.
