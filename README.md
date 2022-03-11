# Polygym

## Getting started
1. Build the Polyite dependency. For this, refer to the steps in https://github.com/PolyGym/polyite. Note that the version of Polly that shall be used is in https://github.com/PolyGym/polly.
2. Change several hard-coded paths in the polygym project to point to your build of Polyite
3. Run a sample exploration with random agents using the `run_random.sh` script (Note that this also requires changing paths to Polyite). This script assumes to run on a AMD 3960X CPU with 24 cores.
