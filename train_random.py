import csv
import numpy as np
import os
import random
import sys
import pandas as pd

from absl import app
from absl import flags

import tqdm

import environment
import polygym
import schedule_eval


flags.DEFINE_string('out_dir', '', 'Root dir to store the results.')
flags.DEFINE_boolean('with_baselines', False, 'Benchmark baselines.')
flags.DEFINE_boolean('with_isl_tuning', False, 'Benchmark isl and tune.')
flags.DEFINE_boolean('with_polyenv', False, 'Benchmark polyenv random walk.')
flags.DEFINE_string('with_polyenv_sampling_bias', None, 'A sampling bias.')

flags.DEFINE_integer('stop_at', None, 'Number of OK samples to stop at.')

flags.DEFINE_string('with_action_import_sample_name', '', 'The sample name of the action sequence to import.')
flags.DEFINE_string('with_action_import_actions', '', 'The action sequence to import.')
FLAGS = flags.FLAGS


def create_csv_if_not_exists(filename, fieldnames):
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()


def append_to_csv(filename, row):
    with open(filename, 'a') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(row)


def gen_and_bench_random_schedule(env, sample_name, sampling_bias=None, predef_actions=None):
    with_ast_and_map = True if sample_name in ['gemm', 'matvect'] else False
    state = env.reset(sample_name, with_ast_and_map=with_ast_and_map)

    actions = []
    done = False
    reward = None

    exec_time = None

    try:
        if predef_actions:
            predef_actions_idx = 0
        while not done:
            if predef_actions:
                action_idx = predef_actions[predef_actions_idx]
                predef_actions_idx += 1
            elif sampling_bias:
                mask = state['action_mask']
                possibilities = mask * range(len(mask))
                if sampling_bias == 'bias_coeff0':
                    p = mask * [1, 1, 1, 1, 0.15, 0.15]
                elif sampling_bias == 'bias_select_dep':
                    p = mask * [0.2, 0.2, 0.6, 1, 1, 1]
                else:
                    raise Exception
                p /= p.sum()        # Normalize
                action_idx = int(np.random.choice(possibilities, p=p))
            else:
                action_idx = np.random.choice(np.nonzero(state['action_mask'])[0])
            action = list(environment.Action)[action_idx]
            actions.append(action_idx)

            state, reward, done, info = env.step(action)

        speedup = env.reward_to_speedup(reward)
        exec_time = env.speedup_to_execution_time(speedup)
        status = info['status']
        ast = info['ast'] if 'ast' in info else None
        isl_map = info['isl_map'] if 'isl_map' in info else None

    except (
            polygym.ChernikovaTimeoutException, schedule_eval.LLVMTimeoutException,
            schedule_eval.InvalidScheduleException,
            schedule_eval.LLVMInternalException, schedule_eval.ScheduleTreeNotLoadedException,
            schedule_eval.OutputValidationException) as e:
        status = e.__class__.__name__

    return exec_time, status, actions, ast if 'ast' in locals() else None, isl_map if 'isl_map' in locals() else None


def bench(invocation, optimization=None, clang_exe=None, additional_params=None):
    sys.argv = invocation
    _, compilation_params, config, scop_file, jsonp, scop = polygym.parse_args()

    if clang_exe:
        config.clang_benchmark_exe = clang_exe

    if optimization == 'O3':
        execution_time = schedule_eval.benchmark_schedule(compilation_params,
                                                          config, scop_file,
                                                          num_iterations=1,
                                                          compilation_timeout=100)
    elif optimization == 'ISL':
        execution_time = schedule_eval.benchmark_schedule(compilation_params,
                                                          config, scop_file,
                                                          with_polly=True,
                                                          num_iterations=1,
                                                          compilation_timeout=100,
                                                          additional_params=additional_params)

    return execution_time, config.clang_benchmark_exe


def main(argv):
    if FLAGS.with_baselines:
        clang_exes = [
                None,
                'clang-10',
                '/tmp/llvm-10/build/bin/clang',
                '/tmp/llvm-12/build/bin/clang',
#                '/tmp/llvm-master/build/bin/clang',
                ]
        optimizations = ['O3', 'ISL']
        for sample_name, invocation in polygym.polybench_invocations.items():
            csv_filename = os.path.join(FLAGS.out_dir, sample_name + '_baselines.csv')
            create_csv_if_not_exists(csv_filename, ['method', 'compiler', 'execution_time'])

            for optimization in optimizations:
                for clang_exe in clang_exes:
                    # Bench
                    exec_time, clang_exe_used = bench(invocation, optimization=optimization, clang_exe=clang_exe)

                    # Save result
                    append_to_csv(csv_filename, [optimization, clang_exe_used, exec_time])
    if FLAGS.with_isl_tuning:
        clang_exe = '/media/local/brauckmann/llvm/llvm-12/build/bin/clang'

        isl_options = {
                'polly-opt-optimize-only': ['all', 'raw'],
                'polly-opt-fusion': ['min', 'max'],
                'polly-opt-max-constant-term': [-1] + list(range(500)),
                'polly-opt-max-coefficient': [-1] + list(range(500)),
                'polly-opt-fusion': ['min', 'max'],
                'polly-opt-maximize-bands': ['yes', 'no'],
                'polly-opt-outer-coincidence': ['yes', 'no'],
                }
        polly_options = {
                'polly-prevect-width': [2**x for x in range(1,6)],
                'polly-target-latency-vector-fma': [2**x for x in range(1,6)],
                'polly-target-throughput-vector-fma': [2**x for x in range(1,4)],
                'polly-default-tile-size': [2**x for x in range(1,7)],
                'polly-register-tiling': [True, False],
                'polly-pattern-matching-based-opts': [True, False]
                }

        for i in range(FLAGS.stop_at):
            for sample_name, invocation in polygym.polybench_invocations.items():
                csv_filename = os.path.join(FLAGS.out_dir, sample_name + '_isl_baselines.csv')
                create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'additional_params'])

                additional_params = []

                for option_name, option_vals in isl_options.items():
                    if random.random() < 0.5:
                        continue

                    option_val = random.choice(option_vals)

                    param = '-' + option_name
                    if type(option_val) is not bool:
                        param += '=' + str(option_val)
                    additional_params.append(param)
                
                # Bench
                try:
                    (exec_time, exception), clang_exe_used = bench(invocation, optimization='ISL', clang_exe=clang_exe, additional_params=additional_params)
                except Exception as e:
                    print(e)
                    continue

                # Save result
                config = ''
                append_to_csv(csv_filename, ['ISL', exec_time, str(additional_params)])

    if FLAGS.with_polyenv:
        env_config = {'invocations': polygym.polybench_invocations}
        env = environment.PolyEnv(env_config)

        to_process = list(polygym.polybench_invocations.keys())

        i = 0
        while True:
            print('to_process: ' + str(to_process))
            print('len(to_process): ' + str(len(to_process)))

            for sample_name in tqdm.tqdm(to_process):
                csv_filename = os.path.join(FLAGS.out_dir, sample_name + '.csv')

                # Remove sample from eval if its evaluated enough already
                if FLAGS.stop_at and os.path.isfile(csv_filename) and i % 1 == 0:
                    df = pd.read_csv(csv_filename, sep="\t")
                    num_ok = len(df[df['status'] == 'ok'])
                    print('sample_name: %s, num_ok: %i' % (sample_name, num_ok))
                    if num_ok > FLAGS.stop_at:
                        to_process.remove(sample_name)
                        print('removed element: ' + sample_name)
                        continue

                # Bench
                exec_time, status, actions, ast, isl_map = gen_and_bench_random_schedule(env, sample_name, FLAGS.with_polyenv_sampling_bias)

                # Save result
                create_csv_if_not_exists(csv_filename, ['method', 'execution_time', 'status', 'actions', 'ast', 'isl_map'])
                append_to_csv(csv_filename, ['PolyEnv-random', exec_time, status, str(actions), str(ast), str(isl_map)])

            i += 1

    if FLAGS.with_action_import_sample_name and FLAGS.with_action_import_actions:
        env_config = {'invocations': polygym.polybench_invocations}
        env = environment.PolyEnv(env_config)

        sample_name = FLAGS.with_action_import_sample_name
        actions = eval(FLAGS.with_action_import_actions)
        exec_time, status, actions, ast, isl_map = gen_and_bench_random_schedule(env, sample_name, False, actions)

        print(isl_map)
        print(ast)
        print(exec_time)

if __name__ == "__main__":
    app.run(main)
