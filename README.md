# Polygym

This repository contains the code of PolyGym, as presented in

```
Brauckmann, Alexander, Andrés Goens, and Jeronimo Castrillon.
"PolyGym: Polyhedral Optimizations as an Environment for Reinforcement Learning."
2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT). IEEE, 2021.
```

## Getting started
1. Build the Polyite dependency. For this, refer to the steps in https://github.com/PolyGym/polyite. Note that the version of Polly that shall be used is in https://github.com/PolyGym/polly.
2. Change several hard-coded paths in the polygym project to point to your build of Polyite.
3. Run a sample exploration with random agents using the `run_random.sh` script (Note that this also requires changing paths to Polyite). This script assumes to run on a AMD 3960X CPU with 24 cores.
