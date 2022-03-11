import enum
import logging
import random
import sys
import time

import gym
import numpy as np
from gym.spaces import Discrete, Box, Dict, Tuple
#from ray.rllib.utils.spaces.repeated import Repeated

#import model
import polygym
#import representation
import schedule_eval


ILLEGAL_ACTION_REWARD = 0
EXECUTION_TIME_CUTOFF_FACTOR = 1000  # Schedules beyond this factor wrt. O3 are terminated


class Action(enum.Enum):
    # Schedule space construction
    next_dim = 0
    next_dep = 1
    select_dep = 2

    # Schedule space exploration
    coeff_0 = 3
    coeff_pos1 = 4
    coeff_pos2 = 5
n_actions = len(list(Action))


class Status(enum.Enum):
    construct_space = 0
    explore_space = 1
    done = 2

MAX_NUM_EDGE_TYPES = 4
MAX_NUM_EDGES = 200
#MAX_NUM_NODE_TYPES = model.ANNOTATION_SIZE
MAX_NUM_NODES = 200

MAX_NUM_DEPS = 10
MAX_NUM_DIMS = 10

REWARD_EXPONENT = 3

class PolyEnv(gym.Env):
    def __init__(self, config):
        self.invocations = config['invocations']
        self.samples = {}

        self.action_space = Discrete(n_actions)

#        graph_space = Dict({
#            'x': Repeated(Discrete(MAX_NUM_NODE_TYPES), max_len=MAX_NUM_NODES),     # node features
#            'edge_index': Tuple([Repeated(Box(low=0, high=MAX_NUM_NODES, shape=[1, 1]), max_len=MAX_NUM_EDGES)] * 2),
#            'edge_features': Repeated(Box(low=0, high=MAX_NUM_EDGE_TYPES, shape=[1, 1]), max_len=MAX_NUM_EDGES)
#        })
#        self.observation_space = Dict({
#            # Status
#            "action_mask": Box(0, 1, shape=[n_actions, 1]),
#
#            # Construct
#            "candidate_dep": graph_space,
#            "available_deps": Repeated(graph_space, max_len=MAX_NUM_DEPS),
#            "selected_deps_by_dim": Repeated(Repeated(graph_space, max_len=MAX_NUM_DEPS), max_len=MAX_NUM_DIMS),
#
#            # Explore
#            'current_coeff_vector': Box(0, 1, shape=[1, 1]),
#            'current_term_candidate_vector': Box(0, 1, shape=[1, 1]),
#            'current_dim_deps': Box(0, 1, shape=[1, 1]),
#            'future_deps_by_dim': Box(0, 1, shape=[1, 1]),
#
#            'previous_deps_by_dim': Box(0, 1, shape=[1, 1]),
#            'previous_coeff_vectors': Box(0, 1, shape=[1, 1]),
#        })

    def _set_sample(self, sample_name):
        if not sample_name:
            print('Selecting random sample')
            sample_name = random.choice(list(self.invocations.keys()))

        print('Using sample with name: ' + sample_name)
        if sample_name not in self.samples:
            # Get invocation
            sys.argv = self.invocations[sample_name]

            # Construct sample
            # - Polyhedral SCoP description
            _, compilation_params, config, scop_file, jsonp, scop = polygym.parse_args()
            deps, dom_info = polygym.calcDepsAndDomInfo(scop.context.params(), scop.domain,
                                                        scop.schedule,
                                                        scop.reads().gist_domain(scop.domain),
                                                        scop.writes().gist_domain(scop.domain))

            # - Representations
#            dep_reps_by_deps = {dep: representation.construct_dependency_graph(compilation_params,
#                                                                               scop_file,
#                                                                               str(dep.baseMap)) for dep in deps}

            # deps = sorted(deps, key=lambda d: (d.tupleNameIn, d.tupleNameOut))

            # - Benchmark O3
            c_code = polygym.printScheduleAsProgram(scop.context.params(),
                                                    polygym.islScheduleMap2IslScheduleTree(jsonp, scop.schedule))
            print('Original Schedule (AST / C Code representation):\n%s' % c_code)
            config = polygym.Config(is_debug=False)

            O3_execution_time, out_array = schedule_eval.benchmark_schedule(compilation_params,
                                                                           config, scop_file,
                                                                           num_iterations=1,
                                                                           with_dumparray=True)
            print('O3 execution time: %.2f' % O3_execution_time)

            # Store in samples
            sample = {
                'compilation_params': compilation_params,
                'config': config,
                'scop_file': scop_file,
                'jsonp': jsonp,
                'scop': scop,
                'deps': deps,
                'dom_info': dom_info,
#                'dep_reps_by_deps': dep_reps_by_deps,
                'O3_execution_time': O3_execution_time,
                'ref_output': out_array
            }
            self.samples[sample_name] = sample

        self.sample = self.samples[sample_name]

    def _filter_state(self, state, with_repr):
        ret = {
            # Status
            'action_mask': state['action_mask']
        }
        if with_repr:
            ret.update({
                # Status
                'action_mask': state['action_mask'],

                # Construct
                'candidate_dep': state['candidate_dep'],
                'available_deps': state['available_deps'],
                'selected_deps_by_dim': state['selected_deps_by_dim'],

                # Explore
                'current_coeff_vector': state['current_coeff_vector'],
                'current_term_candidate_vector': state['current_term_candidate_vector'],
                'current_dim_deps': state['current_dim_deps'],
                'future_deps_by_dim': state['future_deps_by_dim'],

                'previous_deps_by_dim': state['previous_deps_by_dim'],
                'previous_coeff_vectors': state['previous_coeff_vectors'],
            })

        return ret

    def reset(self, sample_name=None, with_repr=False, with_ast_and_map=True):
        self._set_sample(sample_name)

        self.with_ast_and_map = with_ast_and_map

        self.dep_ptr = 0
        self.dim_ptr = 0
        self.term_ptr = 0

        self.schedulePolys = []
        self.schedulePolysDependences = []
        self.schedulePolysDependenceMeta = []

        self.stronglyCarried = []
        self.availableDeps = self.sample['deps'].copy()
        self.uncarriedDeps = set(self.sample['deps'])

        self.coeffs_and_summands_by_dim = []

        self.done_construct = False
        self.done_explore = False
        self.status = Status.construct_space

        self.consecutive_next_dims = 0
        self.consecutive_next_dep_actions = 0

        self.num_steps = 0
        self.last_reward = 0

        self._carry_uncarried_deps_weakly()

        state = {
            # Status
            'action_mask': np.concatenate([np.ones(3), np.zeros(3)]) if self.status == Status.construct_space \
                else np.concatenate([np.zeros(3), np.ones(3)]),
        }
        if with_repr:
            state.update({
                # Status
                'action_mask': np.concatenate([np.ones(3), np.zeros(3)]) if self.status == Status.construct_space \
                    else np.concatenate([np.zeros(3), np.ones(3)]),

                # Construct
                'candidate_dep': self.sample['dep_reps_by_deps'][self.availableDeps[self.dep_ptr]],
                'available_deps': [self.sample['dep_reps_by_deps'][depLeft] for depLeft in self.availableDeps],
                'selected_deps_by_dim': [[]],

                # Explore
                'current_coeff_vector': np.zeros([10]),
                'current_term_candidate_vector': np.zeros([10]),
                'current_dim_deps': [representation.construct_null_dependency_graph()],
                'future_deps_by_dim': [[representation.construct_null_dependency_graph()]],

                'previous_deps_by_dim': [[representation.construct_null_dependency_graph()]],
                'previous_coeff_vectors': [np.zeros([10])],
            })

        # self.MAX_CONSECUTIVE_NEXT_DEP_ACTIONS = len(self.sample['deps']) * 2 + 1
        self.MAX_CONSECUTIVE_NEXT_DIMS = 1

        return self._filter_state(state, with_repr)

    def _skip_vertices(self):
        while True:
            if self._get_term_by_ptr(self.dim_ptr, self.term_ptr)[1] == 'vertex':
                self.dim_ptr, self.term_ptr = self._get_incremented_ptrs(self.dim_ptr, self.term_ptr)
            else:
                break

    def _get_incremented_ptrs(self, dim_ptr, term_ptr):
        gen = self.schedulePolys[dim_ptr]
        if term_ptr < len(gen.vertices) + len(gen.rays) + len(gen.lines) - 1:
            return dim_ptr, term_ptr + 1
        else:
            return dim_ptr + 1, 0

    def _get_term_by_ptr(self, dim_ptr, term_ptr):
        if dim_ptr > len(self.schedulePolys):
            dim_ptr = dim_ptr % self.schedulePolys

        gen = self.schedulePolys[dim_ptr]

        if term_ptr < len(gen.vertices):
            return gen.vertices[term_ptr], 'vertex'
        elif term_ptr < len(gen.vertices) + len(gen.lines):
            return gen.lines[term_ptr - len(gen.vertices)], 'line'
        elif term_ptr < len(gen.vertices) + len(gen.lines) + len(gen.rays):
            return gen.rays[term_ptr - len(gen.vertices) - len(gen.lines)], 'ray'

    def _calc_speedup(self, execution_time):
        return self.sample['O3_execution_time'] / execution_time

    def _carry_uncarried_deps_weakly(self):
        self.currentScheduleDim = self.sample['dom_info'].universe.copy()
        for dependence in self.uncarriedDeps:
            self.currentScheduleDim = self.currentScheduleDim.intersect(dependence.weakConstr)

    def _can_dep_be_carried_strongly(self, dependence):
        return not self.currentScheduleDim.intersect(dependence.strongConstr).is_empty()

    def _add_polyhedron(self):
        # then, try to strongly carry selected dependences
        carriedInCurrentDim = set()
        for dependence in self.stronglyCarried:
            currentScheduleDimMaybe = self.currentScheduleDim.intersect(dependence.strongConstr)
            # only carry if possible in this dimension
            if not currentScheduleDimMaybe.is_empty():
                carriedInCurrentDim.add(dependence)
                self.currentScheduleDim = currentScheduleDimMaybe
        self.uncarriedDeps -= carriedInCurrentDim
        self.availableDeps = list(self.uncarriedDeps)

        polyhedron = polygym.preparePolyhedron(self.currentScheduleDim)
        polyhedron = polygym.transformGeneratorInfoLinesToRays(polyhedron)

        self.schedulePolys.append(polyhedron)
        self.schedulePolysDependences.append(self.stronglyCarried)
        self.schedulePolysDependenceMeta.append([(x.tupleNameIn, x.tupleNameOut) for x in carriedInCurrentDim])
        print(self.schedulePolysDependenceMeta)

        if len(carriedInCurrentDim) == 0:
            self.consecutive_next_dims += 1

        # prepare next iteration
        self.stronglyCarried.clear()
        self._carry_uncarried_deps_weakly()

    def step(self, action, with_repr=False):
        self.num_steps += 1

        # Map to enum
        action = Action(action)

        logging.debug('Action: %s' % str(action))

        info = {
            'status': 'ok'
        }
        state = {
            # Status
            'action_mask': np.concatenate([np.ones(3), np.zeros(3)]) if self.status == Status.construct_space \
                            else np.concatenate([np.zeros(3), np.ones(3)]),
        }
        if with_repr:
            state.update({
                # Status
                'action_mask': np.concatenate([np.ones(3), np.zeros(3)]) if self.status == Status.construct_space \
                    else np.concatenate([np.zeros(3), np.ones(3)]),

                # Construct
                'candidate_dep': representation.construct_null_dependency_graph(),
                'available_deps': [representation.construct_null_dependency_graph()],
                'selected_deps_by_dim': [[]],

                # Explore
                'current_coeff_vector': np.zeros([10]),
                'current_term_candidate_vector': np.zeros([10]),
                'current_dim_deps': [representation.construct_null_dependency_graph()],
                'future_deps_by_dim': [[representation.construct_null_dependency_graph()]],

                'previous_deps_by_dim': [[representation.construct_null_dependency_graph()]],
                'previous_coeff_vectors': [np.zeros([10])],
            })

        # Construct
        if action in [Action.next_dim, Action.next_dep, Action.select_dep]:
            # Polyhedra construction
            if self.status != Status.construct_space:
                raise Exception

            if action == Action.next_dim:
                self._add_polyhedron()
                self.dim_ptr += 1
                self.dep_ptr = 0

                self._complete_construct_maybe()

            elif action == Action.next_dep:
                self.dep_ptr += 1
                if self.dep_ptr >= len(self.availableDeps):
                    self.dep_ptr = 0

            elif action == Action.select_dep:
                dep = self.availableDeps[self.dep_ptr]
                self.stronglyCarried.append(dep)

                del self.availableDeps[self.dep_ptr]

                if len(self.availableDeps) == 0:
                    self._complete_construct_maybe()

                if self.dep_ptr >= len(self.availableDeps):
                    self.dep_ptr = 0

            # State
            if with_repr:
                if len(self.availableDeps) > 0:
                    state['candidate_dep'] = self.sample['dep_reps_by_deps'][self.availableDeps[self.dep_ptr]]

                available_deps = [self.sample['dep_reps_by_deps'][dep] for dep in self.availableDeps if
                             self.availableDeps[self.dep_ptr] != dep]
                if len(available_deps) > 0:
                    state['available_deps'] = available_deps

                if len(self.schedulePolysDependences) > 0:
                    state['selected_deps_by_dim'] = [[self.sample['dep_reps_by_deps'][dep] for dep in dim] for dim in self.schedulePolysDependences]

            reward = 0

        # Explore
        elif action in [Action.coeff_0, Action.coeff_pos1, Action.coeff_pos2]:
            # Check for illegal actions
            if self.status != Status.explore_space:
                raise Exception

            # Decode coefficient
            if action == Action.coeff_0:
                coeff = 0
            elif action == Action.coeff_pos1:
                coeff = 1
            elif action == Action.coeff_pos2:
                coeff = 2

            if action == Action.coeff_0:
                reward = 0  # self.last_reward
            else:
                # Get summand
                term, term_type = self._get_term_by_ptr(self.dim_ptr, self.term_ptr)
                summand = None
                if term_type == 'line':
                    summand = polygym.LineSummand(term, coeff)
                elif term_type == 'ray':
                    if coeff < 0:
                        coeff *= -1
                    summand = polygym.RaySummand(term, coeff)

                # Add to current status
                if self.dim_ptr >= len(self.coeffs_and_summands_by_dim):
                    self.coeffs_and_summands_by_dim.append([])
                self.coeffs_and_summands_by_dim[self.dim_ptr].append((coeff, summand))

                reward = 0

            # Update ptrs for next step
            self.dim_ptr, self.term_ptr = self._get_incremented_ptrs(self.dim_ptr, self.term_ptr)
            done_explore = self.dim_ptr >= len(self.schedulePolys)
            if done_explore:
                self._create_schedule()
                if self.with_ast_and_map:
                    info['ast'] = self._get_current_schedule_as_ast()
                    info['isl_map'] = self._get_current_schedule_as_map()
                reward, info['status'] = self._benchmark_current_schedule()
                state_filtered = self._filter_state(state, with_repr)
                return state_filtered, reward, True, info
            self._skip_vertices()

            # State
            if with_repr:
                coeff_vectors = [self._summands_to_coeff_vector([summand for _, summand in coeffs_and_summands]) for
                                 coeffs_and_summands in self.coeffs_and_summands_by_dim]
                state['current_coeff_vector'] = coeff_vectors[self.dim_ptr]

                state['current_term_candidate_vector'] = self._get_term_by_ptr(self.dim_ptr, self.term_ptr)[0]

                previous_deps_by_dim = self.schedulePolysDependences[:self.dim_ptr]
                if len(previous_deps_by_dim) > 0:
                    state['previous_deps_by_dim'] = [[self.sample['dep_reps_by_deps'][dep] for dep in dim] for dim in previous_deps_by_dim]

                previous_coeff_vectors = coeff_vectors[:self.dim_ptr]
                if len(previous_coeff_vectors) > 0:
                    state['previous_coeff_vectors'] = previous_coeff_vectors

                next_terms = []
                for generator in self.schedulePolys:
                    for line in generator.lines:
                        next_terms.append(line)
                    for ray in generator.rays:
                        next_terms.append(ray)
                state['next_terms'] = next_terms

                current_dim_deps = [self.sample['dep_reps_by_deps'][dep] for dep in self.schedulePolysDependences[self.dim_ptr]]
                if len(current_dim_deps) > 0:
                    state['current_dim_deps'] = current_dim_deps

                if (self.dim_ptr + 1) < len(self.schedulePolys):
                    state['future_deps_by_dim'] = [[self.sample['dep_reps_by_deps'][dep] for dep in dim] for dim in self.schedulePolysDependences[self.dim_ptr+1:]]

        # Update action mask
        state['action_mask'] = np.concatenate([np.ones(3), np.zeros(3)]) if self.status == Status.construct_space \
            else np.concatenate([np.zeros(3), np.ones(3)])

        # Limit actions
        if action == Action.next_dim:
            self.consecutive_next_dims += 1
        elif action == Action.select_dep:
            self.consecutive_next_dims = 0
        if self.consecutive_next_dims >= self.MAX_CONSECUTIVE_NEXT_DIMS:
            state['action_mask'][Action.next_dim.value] = 0

        # self.consecutive_next_dep_actions = self.consecutive_next_dep_actions + 1 if action == Action.next_dep else 0
        # if self.consecutive_next_dep_actions > self.MAX_CONSECUTIVE_NEXT_DEP_ACTIONS:
        #     state['action_mask'][Action.next_dep.value] = 0

        if state['action_mask'][Action.select_dep.value] == 1 and len(self.availableDeps) > 0 and not self._can_dep_be_carried_strongly(self.availableDeps[self.dep_ptr]):
            state['action_mask'][Action.select_dep.value] = 0

        state_filtered = self._filter_state(state, with_repr)

        return state_filtered, reward, False, info

    def _complete_construct_maybe(self):
        self._add_polyhedron()

        if len(self.uncarriedDeps) == 0:
            self.status = Status.explore_space
            self.dim_ptr = 0

            print('\nDone constructing schedule space:\n%s\n' % self.schedulePolysDependenceMeta)

            # Every dimension needs a vertex
            for region in self.schedulePolys:
                # HACK
                vertex = None
                if len(region.vertices) == 1:
                    vertex = region.vertices[0]
                else:
                    print('WARNING: More than 1 vertex!')
                    vertex = random.choice(region.vertices)

                self.coeffs_and_summands_by_dim.append([(1, polygym.VertexSummand(vertex, 1))])
            self._skip_vertices()

    def _summands_to_coeff_vector(self, summands):
        coeff_vector = summands[0].getValue()
        for j, summand in enumerate(summands):
            if j == 0:
                continue
            coeff_vector = polygym.add(coeff_vector, summand.getValue(), 1)

        coeff_vector = polygym.multiplyWithCommonDenominator(coeff_vector)

        return coeff_vector

    def _create_schedule(self):
        sched = polygym.Schedule(self.sample['dom_info'], self.availableDeps)
        for i, coeffs_and_summands in enumerate(self.coeffs_and_summands_by_dim):
            summands = [coeffs_and_summands[i][1] for i in range(len(coeffs_and_summands))]

            coeff_vector = self._summands_to_coeff_vector(summands)

            sched.addScheduleVector(coeff_vector, summands)

#        print(sched)
        # Integerize schedule vectors
        sched.scheduleVectors = [polygym.multiplyWithCommonDenominator(x) for x in sched.scheduleVectors]

        self.sched = sched

    def _benchmark_current_schedule(self):
        print('Benchmarking schedule...')
        # Benchmark
        # - Execute
        isl_schedule_map = polygym.coeffMatrix2IslUnionMap(self.sample['dom_info'], self.sched.scheduleVectors)
        print(isl_schedule_map)
        isl_schedule_tree = polygym.islScheduleMap2IslScheduleTree(self.sample['jsonp'], isl_schedule_map)
        execution_time, out_array = schedule_eval.benchmark_schedule(self.sample['compilation_params'], self.sample['config'], self.sample['scop_file'],
                                                                     schedule_tree=isl_schedule_tree, num_iterations=1,
                                                                     benchmark_timeout=self.sample['O3_execution_time'] * EXECUTION_TIME_CUTOFF_FACTOR,
                                                                     with_dumparray=True)

        if execution_time:
            reward = self._calc_speedup(execution_time)
            status = 'ok'
            print('└ PolyEnv Execution time: %.3f, Reward: %.3f' % (execution_time, reward))
        else:
            reward = 1 / EXECUTION_TIME_CUTOFF_FACTOR
            status = 'benchmark_timeout'
            logging.info('└ Execution takes too long. Reward: %.3f' % reward)

        if out_array:
            start = time.time()
            if self.sample['ref_output'] != out_array:
                raise schedule_eval.OutputValidationException
            end = time.time()
            print('Ref output validation took: ', end - start)

        reward = reward ** REWARD_EXPONENT

        return reward, status

    def _get_current_schedule_as_ast(self):
        isl_schedule_map = polygym.coeffMatrix2IslUnionMap(self.sample['dom_info'], self.sched.scheduleVectors)
        isl_schedule_tree = polygym.islScheduleMap2IslScheduleTree(self.sample['jsonp'], isl_schedule_map)
        c_code = polygym.printScheduleAsProgram(self.sample['scop'].context.params(), isl_schedule_tree)

        return c_code

    def _get_current_schedule_as_map(self):
        isl_schedule_map = polygym.coeffMatrix2IslUnionMap(self.sample['dom_info'], self.sched.scheduleVectors)

        return isl_schedule_map

    def reward_to_speedup(self, reward):
        return reward ** (1/float(REWARD_EXPONENT))

    def speedup_to_execution_time(self, speedup):
        return self.sample['O3_execution_time'] / speedup
