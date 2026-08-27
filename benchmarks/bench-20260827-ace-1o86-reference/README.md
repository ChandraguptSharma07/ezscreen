# ACE/1O86 fast reference screen

This is SwiftScreen's small, outcome-known positive-control benchmark. It uses
1,846 DUD Release 2 records (49 actives and 1,797 decoys) and the pipeline's
frozen AutoDockTools-first preparation contract. All 1,846 ligands were prepared
with AutoDockTools, all were scored, and no fallback was used.

The final raw UniDock/Vina ranking reproduces the published AutoDock Vina values
from Empereur-Mot et al. (2015): EF2 is 3.054606 versus 3.05 published, and EF5
is 2.835637 versus 2.84 published. Both match when rounded to the publication's
two-decimal precision.

Run the fail-closed verification with:

```bash
.venv/bin/python benchmarks/bench-20260827-ace-1o86-reference/verify_reference.py
```

The benchmark is suitable as a preparation/ranking regression test. It is not
an untouched validation target and must not be used to justify changes selected
after inspecting ACE outcomes. See `protocol.json` for the exact contract and
comparison limitations.
