python ./code/ternary_train.py --config ./config/policy_briefing_config.yaml
wait
python ./code/ternary_inference.py --config ./config/policy_briefing_config.yaml
wait

python ./code/ternary_train.py --config ./config/wikipedia_config.yaml
wait
python ./code/ternary_inference.py --config ./config/wikipedia_config.yaml
wait

python ./code/ternary_train.py --config ./config/wikitree_config.yaml
wait
python ./code/ternary_inference.py --config ./config/wikitree_config.yaml
wait 