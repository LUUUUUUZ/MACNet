python run.py \
  --model MACNet \
  --embedding-dim 128 \
  --dataset-dir ./datasets/delicious \
  --weight-decay 0.01\
  --lr 1e-3 \
  --batch-size 256 \
  --device 4 \
  --epochs 40 \
  --log-interval 500 \
  --num-cluster 4000 \
  --alpha 0.1 \
  --beta 0.2 \
  --augment-type 'reorder' \