# ASP Optimisation
This directory contains ASP programs to find the best interpretation of the
conjuncts (`a_{i}`) used by the Neural DNF based agents in `DoorCorridor` env.

`context_generator.py` is incorrect at the moment.

**TODO**: At each time, check the action. Ensure the image encoding(s) needed is
fired for that action. So change the definition of `fired_img_encoding`

## Usage

Generate a context program based on the rules extracted from the model. Here
there are 3 context programs based on rules extracted from 3 different models.

Then run clingo to find all optimal models:

```bash
clingo base.lp <Context file> --opt-mode=optN
```

Note that `--opt-mode=optN` is crucial as it enumerates all optimal models.
