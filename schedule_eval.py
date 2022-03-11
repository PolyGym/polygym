import logging
import json
import os
import re
import tempfile
import time
import tqdm
import subprocess


class LLVMTimeoutException(Exception):
    pass


class LLVMInternalException(Exception):
    pass


class InvalidScheduleException(Exception):
    pass


class ScheduleTreeNotLoadedException(Exception):
    pass


class OutputValidationException(Exception):
    pass


polybench_defines = ["-DPOLYBENCH_USE_C99_PROTO",
                     "-DPOLYBENCH_TIME"]


def extract_jscop(compilation_params, config):
    tmp_dir = tempfile.mkdtemp()

    cmd = [config.clang_benchmark_exe,
           "-O3",
           "-mllvm -polly",
           "-mllvm -polly-export"]
    if compilation_params['function_name'] == 'kernel_gramschmidt':
        cmd += ["-mllvm -polly-process-unprofitable=true"]
    cmd += ["-mllvm -polly-only-func=%s" % compilation_params['function_name'],
           "-mllvm -polly-dependences-computeout=0",
           "-mllvm -polly-optimizer=none",
           "-lm"]
    cmd += polybench_defines

    for include_dir in compilation_params['include_dirs']:
        cmd += ['-I ' + include_dir]
    for source_file in compilation_params['source_files']:
        cmd += [source_file]

    cmd += ["-o /dev/null"]

    print(tmp_dir)
    print(' '.join(cmd))

    out = subprocess.check_output(' '.join(cmd).split(' '), cwd=tmp_dir)
    out = out.decode('utf-8')
    print("extract_jscop clang out: %s" % out)

    files = os.listdir(tmp_dir)
    assert len(files) >= 1

    num_scops = len(files)
    if num_scops == 1:
        scop_file = files[0]
    else:
        file_sizes = [os.path.getsize(os.path.join(tmp_dir, f)) for f in files]
        max_file_size = max(file_sizes)

        scop_file = files[file_sizes.index(max_file_size)]

    region_lhs = re.search('%(.*?)-', scop_file).group(1)
    region_rhs = re.search('--%(.*?).jscop', scop_file).group(1)
    region = '%' + region_lhs + ' => ' + '%' + region_rhs

    return os.path.join(tmp_dir, scop_file), region, num_scops


def benchmark_schedule(compilation_params, config, scop_file, schedule_tree=None, num_iterations=5, compilation_timeout=10,
                       benchmark_timeout=None, with_polly=False, with_dumparray=False, additional_params=None):
    if schedule_tree:
        exe_name = "a.out"
    else:
        exe_name = "a_O3.out"

    # Write JSCOP
    if schedule_tree:
        print('scop_file', scop_file)
        with open(scop_file) as file:
            jsonp = json.load(file)

        jsonp["schedTree"] = str(schedule_tree).replace("\"", "\"")

        with open(scop_file, "w") as file:
            json.dump(jsonp, file, indent=2, sort_keys=True)

    # Compile
    cmd = ["timeout", "%s" % compilation_timeout,
           config.clang_benchmark_exe,
           "-O3"]
    if schedule_tree or with_polly:
        cmd += ["-mllvm", "-polly"]
    if schedule_tree:
        cmd += ["-mllvm", "-polly-import",
                "-mllvm", "-polly-optimizer=none",
                "-mllvm", "-polly-dependences-computeout=0",
                "-mllvm", "-polly-import-jscop-read-schedule-tree=true",
                "-mllvm", "-polly-only-func=%s" % compilation_params['function_name']]
    if schedule_tree or with_polly:
        if compilation_params['num_scops'] > 1:
            cmd += ["-mllvm", "-polly-only-region='%s'" % compilation_params['region']]
        cmd += ["-mllvm", "-polly-position=early"]
        if compilation_params['function_name'] == 'kernel_gramschmidt':
            cmd += ["-mllvm -polly-process-unprofitable=true"]
        cmd += ["-mllvm", "-polly-parallel=true",
                "-mllvm", "-polly-dependences-computeout=0",
                "-mllvm", "-polly-tiling=true",
                "-mllvm", "-polly-default-tile-size=64",
                "-mllvm", "-polly-vectorizer=none"
        ]
    if additional_params:
        for param in additional_params:
            cmd += ["-mllvm", param]

    cmd += polybench_defines
    if with_dumparray:
        cmd += ["-DPOLYBENCH_DUMP_ARRAYS"]

    for include_dir in compilation_params['include_dirs']:
        cmd += ['-I', include_dir]
    for source_file in compilation_params['source_files']:
        cmd += [source_file]

    cmd += ["-lm", "-lgomp", "-o", exe_name]

    print(os.path.dirname(scop_file))
    print(' '.join(cmd))

    try:
        out = subprocess.check_output(' '.join(cmd), cwd=os.path.dirname(scop_file), stderr=subprocess.STDOUT, shell=True, universal_newlines=True) #, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("LLVM error! ", e.returncode, e.output)

        if e.returncode == 124:                                       # This is the error code for timed out execution
            raise LLVMTimeoutException
        elif schedule_tree and 'JScop file contains a schedule that changes the dependences' in e.output:
            raise InvalidScheduleException
        else:
            raise LLVMInternalException

    if schedule_tree and 'Loading schedule tree' not in out:
        print("Polly error! Schedule could not be loaded!")
        raise ScheduleTreeNotLoadedException

    print('benchmark_schedule clang out: %s' % out)

    # Run and measure
    #start_time = time.time()
    cmd = ['./' + exe_name]
    if benchmark_timeout:
        cmd = ['timeout', '%s' % benchmark_timeout] + cmd

    print(os.path.dirname(scop_file))
    print(' '.join(cmd))

    process = subprocess.Popen(cmd, cwd=os.path.dirname(scop_file), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode == 124:   # This is the error code for timed out execution
        return None, None
    if process.returncode != 0:
        print("benchmark error. code: ", process.returncode, out, err)
        return None, None

    execution_time = float(out.decode('ascii').replace('\n', ''))
    if with_dumparray:
        dumparray = err.decode('ascii')
        return execution_time, dumparray

    return execution_time, None
