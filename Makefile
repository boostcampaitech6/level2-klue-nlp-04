run: train inference
clean: delete-checkpoints delete-best_model
clean-run: clean run
setup: set-precommit set-style-dep #set-git
style: set-style-dep set-style
quality: set-style-dep check-quality

##### run #####
train:
	python ./code/train.py

inference:
	python ./code/inference.py

all: 
	./read_config.sh

##### clean #####
delete-checkpoints:
	rm -rf ./results/*

delete-best_model:
	rm -rf ./best_model/*

##### setup #####
# 작동 안됨: 원인 미상
# set-git:
# 	git config --local commit.template .gitmessage

set-precommit:
	pip3 install pre-commit==2.17.0
	pre-commit install

set-style-dep:
	pip3 install isort==5.12.0 black==23.3.0 flake8==4.0.1

##### style ######
set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .

##### quality #####
check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

# Reference:
#	https://github.com/monologg/python-template