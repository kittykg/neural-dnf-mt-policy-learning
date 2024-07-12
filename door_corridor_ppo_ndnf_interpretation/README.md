# ASP Optimisation
This directory contains ASP programs to find the best interpretation of the
conjuncts (`a_{i}`) used by the Neural DNF based agents in `DoorCorridor` env.
The interpretation of `a_{i}` is a conjunction of observations.

`context_generator.py` generates the context program based on each model and its
ASP rules. `interpreter.py` uses clingo to find the best interpretation of each
bit of the image encoding.

`base.lp` contains the common ASP rules for all context programs.

We use clingo to find the interpretation by combining the context program and
the base program and find the stable models. Each stable model gives a set of
observations to use for each image encoding used by the model.

To key idea is:

```bash
clingo base.lp <Context file> --opt-mode=optN
```
`--opt-mode=optN` is crucial as it enumerates all optimal models.

Note that not every model has a stable model. This might be some bug in the ASP
program or the model itself. We are still investigating this.

## Usage

**Pre-requisite**

The model should have gone through training and post-training process, i.e.
`door_corridor_ppo_ndnf_mt_multirun_eval.py` has been run. A eval config file
should be present in `configs/eval/` directory.

**Step 1**

Generate a context program for each model with the same eval config used for
the post-training process. For example:

```bash
python context_generator.py +eval=door_corridor_ppo_ndnf_mt_multirun_eval
```

**Step 2**

Run clingo to find the best interpretation using the context files generated in
the previous step. The eval config is the same as the previous step. For
example:

```bash
python interpreter.py +eval=door_corridor_ppo_ndnf_mt_multirun_eval
```

## Output

A `interpret_result.json` is generated and saved in the same directory as the
model. If there are stable models, the file will contain all the optimal stable
models and its 'translation'. 
