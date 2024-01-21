for config in ./config/config*
do
  python ./code/train.py --config $config
  wait
  python ./code/inference.py --config $config
  wait
done