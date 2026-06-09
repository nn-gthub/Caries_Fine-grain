#!/bin/bash
# Run once to create tooth_crops/ flat folder
mkdir -p /data1/neena/finegrain_alpha_experiments/tooth_crops

# FG-annotated crops
cp /Nasbackup/lab_nirmal/neena/datasets/Alphadent/finegrain_alpha/annot_crops_combined_360/*.jpg /data1/neena/finegrain_alpha_experiments/tooth_crops/ 2>/dev/null
cp /Nasbackup/lab_nirmal/neena/datasets/Alphadent/finegrain_alpha/annot_crops_combined_360/*.png /data1/neena/finegrain_alpha_experiments/tooth_crops/ 2>/dev/null

# All other crops from v2 (train + val)
for split in train val; do
  for cls in caries decolor normal pre-caries; do
    SRC="/Nasbackup/lab_nirmal/neena/datasets/Alphadent/tooth_crops_final_v2/$split/$cls"
    if [ -d "$SRC/images" ]; then
      cp "$SRC/images"/*.jpg /data1/neena/finegrain_alpha_experiments/tooth_crops/ 2>/dev/null
    elif [ -d "$SRC" ]; then
      cp "$SRC"/*.jpg /data1/neena/finegrain_alpha_experiments/tooth_crops/ 2>/dev/null
    fi
  done
done

# Test crops
for cls in caries decolor normal pre-caries; do
  cp /Nasbackup/lab_nirmal/neena/datasets/Alphadent/tooth_crops_test/cls/$cls/*.jpg /data1/neena/finegrain_alpha_experiments/tooth_crops/ 2>/dev/null
done

echo "Total crops: $(ls /data1/neena/finegrain_alpha_experiments/tooth_crops/ | wc -l)"
