rm -rf /tmp/procs_out
mkdir /tmp/procs_out

# PolyEnv
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 0-5   python examples/poly-rl/train_random.py --out_dir /tmp/out_bias_select_dep --with_polyenv --with_polyenv_stop_at 1000 --with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/0.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 6-11  python examples/poly-rl/train_random.py --out_dir /tmp/out_bias_select_dep --with_polyenv --with_polyenv_stop_at 1000 --with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/1.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 12-17 python examples/poly-rl/train_random.py --out_dir /tmp/out_bias_select_dep --with_polyenv --with_polyenv_stop_at 1000 --with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/2.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 18-23 python examples/poly-rl/train_random.py --out_dir /tmp/out_bias_select_dep --with_polyenv --with_polyenv_stop_at 1000 --with_polyenv_sampling_bias bias_select_dep >> /tmp/procs_out/3.out 2>&1 &
wait

# Baselines
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 0-5   python examples/poly-rl/train_random.py --out_dir /tmp/out --with_baselines >> /tmp/procs_out/0.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 6-11  python examples/poly-rl/train_random.py --out_dir /tmp/out --with_baselines >> /tmp/procs_out/1.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 12-17 python examples/poly-rl/train_random.py --out_dir /tmp/out --with_baselines >> /tmp/procs_out/2.out 2>&1 &
POLYITEDIR=/net/home/brauckmann/poly/polyite LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/home/brauckmann/poly/polyite/scala-isl-utils/libs OMP_NUM_THREADS=6 PYTHONPATH=/net/home/brauckmann/poly/compy-learn-polyite taskset -c 18-23 python examples/poly-rl/train_random.py --out_dir /tmp/out --with_baselines >> /tmp/procs_out/3.out 2>&1 &
wait
