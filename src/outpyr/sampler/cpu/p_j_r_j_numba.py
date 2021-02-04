import os
import shutil
import datetime
import pickle

import numpy as np
import tables

import outpyr.helpers as h
import outpyr.helpers_numba as hn


class PjRjNumbaCPUSampler:
    def __init__(self, id_, seed, settings, data, first_gene, first_sample):
        self.id = id_
        self.seed = seed
        self.s = settings
        if len(data.shape) == 1:
            data = np.reshape(data, (1, len(data)))
        self.data = data
        self.first_gene = first_gene
        self.first_sample = first_sample



        self.J = self.data.shape[0]


        self.N = self.data.shape[1]







        self.history_j = {







            'p': [],
            'r': [],




        }
        self.previous_save_time = None




















        self.v_fname = 'variables.pickle'

        if os.path.isfile(self.v_fname):
            self.print(
                "Previous sampling data found in current directory. Loading it...")
            with open(self.v_fname, 'rb') as f:
                self.v = pickle.load(f)
                self.prng = np.random.RandomState(self.seed)
                self.prng.set_state(self.v['numpy_random_state'])
        else:
            self.print(
                "No previous sampling data found, sampling from scratch..."
            )
            self.prng = np.random.RandomState(self.seed)
            self.v = {
                'numpy_random_state': self.prng.get_state(),

                'a_j_array': np.reshape(self.s.A0 + self.data.sum(axis=1, dtype=np.float64), (self.J, 1)),
                'p_j': None,
                'r_j': None,












                'p_j_cumulative': np.zeros((self.data.shape[0], 1)),
                'r_j_cumulative': np.zeros((self.data.shape[0], 1)),







                'r_avg_diff_matrix': np.zeros(self.data.shape),
                'pr_avg_diff_matrix': np.zeros(self.data.shape),



                'am_delta_matrix': np.zeros(self.data.shape),
                'log_gm_delta_matrix': np.zeros(self.data.shape),
                'd_kl_matrix': np.zeros(self.data.shape),

                'am_delta_normalized_matrix': np.zeros(self.data.shape),
                'log_gm_delta_normalized_matrix': np.zeros(self.data.shape),
                'd_kl_normalized_matrix': np.zeros(self.data.shape),

                'time_delta_total': datetime.timedelta(),
                'time_delta_total_warmup': datetime.timedelta(),
                'time_delta_total_post_warmup': datetime.timedelta(),
                'time_delta_iteration': datetime.timedelta(),
                'time_delta_gene': datetime.timedelta(),

                'iteration': 0,
                'SAVE_TRACE': self.s.SAVE_TRACE,
                'SAVE_WARMUP': self.s.SAVE_WARMUP,

                'j': 0,
                'e_j_incomplete': None,
                'e_ji_matrix_incomplete': None,
            }

        self.v['ITERATIONS'] = self.s.ITERATIONS
        self.v['WARMUP'] = self.s.WARMUP





        if self.v['iteration'] == 0:


            self.print('Defining conjugate priors...')
















            _a_j_array = np.repeat(self.s.A0, self.J)
            _a_j_array.shape = (self.J, 1)
            _b_j_array = np.repeat(self.s.B0, self.J)
            _b_j_array.shape = (self.J, 1)
            _e_j_array = np.repeat(self.s.E0, self.J)
            _e_j_array.shape = (self.J, 1)
            _f_j_array = np.repeat(self.s.F0, self.J)
            _f_j_array.shape = (self.J, 1)
            self.v['p_j'] = self.prng.beta(_a_j_array, _b_j_array)
            self.v['r_j'] = self.prng.gamma(_e_j_array, np.reciprocal(_f_j_array))





            if self.should_save_history():







                self.history_j['p'].append(self.v['p_j'])
                self.history_j['r'].append(self.v['r_j'])










    def print(self, *args, **kwargs):
        return h.myprint(*args, **kwargs)

    def delete_print_lines(self, n):
        h.delete_print_lines(n)

    def run(self):
        t1 = h.now()
        self.print('Start train run', h.format_time(t1))
        self.print("Iterative posterior updating...")
        self.print("=========================")

        self.sample(self.s.ITERATIONS + self.s.WARMUP - self.v['iteration'])
        self.post_sample()

    def sample(self, iterations):
        s = self.s





        a_ji_matrix = self.s.A0 + self.data

        _t_iter_begin = h.now()




        self.previous_save_time = _t_iter_begin

        _a_j_array = self.v['a_j_array']

        _e_j_array = np.zeros((self.J, 1))
        _e_ji_matrix = np.zeros(self.data.shape)
        _first_iteration = self.v['iteration']
        for _iteration in range(self.v['iteration'], self.v['iteration'] + iterations):
            self.print('=========================')
            self.print('Starting iteration %d/%d(%d + %d)' % (self.v['iteration'] + 1, self.s.WARMUP + self.s.ITERATIONS, self.s.WARMUP, self.s.ITERATIONS))
            self.print('=========================')




















            p_j = self.v['p_j']
            r_j = self.v['r_j']

            _b_j_array = (r_j * self.N) + s.B0



            r_j.shape = (self.J, 1)
            N_vector = np.repeat(self.N, self.N)
            N_vector.shape = (1, self.N)
            _b_ji_matrix = (r_j * N_vector) + s.B0



            _f_j_array = np.negative(p_j) + 1
            np.log(_f_j_array, _f_j_array)
            _f_j_array *= self.N
            np.negative(_f_j_array, _f_j_array)
            _f_j_array += s.F0
            np.reciprocal(_f_j_array, _f_j_array)

            _f_j_column = _f_j_array.copy()
            _f_j_column.shape = (self.J, 1)
            _f_ji_matrix = np.tile(_f_j_column, self.N)

            _e_j_array, _e_ji_matrix = hn.sync_prng_state(self.prng)(hn.crt_p_j_r_j)(self.data, self.J, self.N, r_j, self.s.E0, _e_j_array, _e_ji_matrix)





            self.v['p_j'] = self.prng.beta(
                _a_j_array,
                _b_j_array,
            )
            self.v['r_j'] = self.prng.gamma(
                _e_j_array,
                _f_j_array,
            )





            if self.should_save_history():







                self.history_j['p'].append(self.v['p_j'])
                self.history_j['r'].append(self.v['r_j'])





            _p_j = self.v['p_j'].copy()
            _p_j.shape = (self.J, 1)

            _r_j = self.v['r_j'].copy()
            _r_j.shape = (self.J, 1)

            if _iteration >= self.s.WARMUP:







                self.v['p_j_cumulative'] += _p_j
                self.v['r_j_cumulative'] += _r_j





            if _iteration >= self.s.SCORE_WARMUP:
                _b_j_array.shape = (self.J, 1)
                _e_j_array.shape = (self.J, 1)
                _f_j_array.shape = (self.J, 1)














                p_matrix = self.prng.beta(
                    a_ji_matrix,
                    _b_ji_matrix,
                )








                r_matrix = self.prng.gamma(
                    _e_ji_matrix,
                    _f_ji_matrix,
                )

                r_diff_matrix = np.abs(r_matrix - _r_j)
                self.v['r_avg_diff_matrix'] += r_diff_matrix

                self.v['pr_avg_diff_matrix'] += ((p_matrix - _p_j) ** 2 + (r_matrix - _r_j) ** 2) / (_p_j ** 2 + _r_j ** 2)

                if (_iteration - self.s.SCORE_WARMUP) % self.s.SCORE_SAVE_INTERVAL == 0:
                    for score in [
                        'r_avg_diff_matrix',
                        'pr_avg_diff_matrix'
                    ]:
                        pass










































            _t_iter_end = h.now()
            _t_iter_delta = _t_iter_end - _t_iter_begin
            _t_iter_begin = _t_iter_end

            self.v['time_delta_total'] += _t_iter_delta

            if _iteration < self.s.WARMUP:
                _phase = 'warmup'
                self.v['time_delta_total_warmup'] += _t_iter_delta
                self.v['time_delta_iteration_warmup'] = _t_iter_delta
            else:
                _phase = 'post-warmup'
                self.v['time_delta_total_post_warmup'] += _t_iter_delta
                self.v['time_delta_iteration_post_warmup'] = _t_iter_delta

            self.v['iteration'] = _iteration + 1

            if _iteration == _first_iteration:
                self.delete_print_lines(3)
            else:
                self.delete_print_lines(8)
            self.print('Iteration %d/%d (%s phase) took' % (_iteration + 1, (self.s.ITERATIONS + self.s.WARMUP), _phase), _t_iter_delta)
            self.print('Total time till now', self.v['time_delta_total'])
            if (_iteration + 1) < self.s.WARMUP:
                total_time_estimated_warmup = self.v['time_delta_total_warmup'] / (_iteration + 1) * self.s.WARMUP
                self.print('Estimated total warmup time remaining', total_time_estimated_warmup - self.v['time_delta_total_warmup'])
                self.print('Estimated total warmup time', total_time_estimated_warmup)
                self.print('Warning: still in warmup phase, post warmup time-length estimation is not available.')
            else:


                total_time_estimated_warmup = self.v['time_delta_total_warmup']
                self.print('Warmup took', total_time_estimated_warmup)
                if (_iteration + 1) > self.s.WARMUP:
                    total_time_estimated_post_warmup = self.v['time_delta_total_post_warmup'] / ((_iteration + 1) - self.s.WARMUP) * self.s.ITERATIONS
                    total_time_estimated = total_time_estimated_warmup + total_time_estimated_post_warmup
                    if self.v['iteration'] < (self.s.ITERATIONS + self.s.WARMUP):
                        self.print('Post-warmup time till now', self.v['time_delta_total_post_warmup'])
                        self.print('Remaining time %s/%s' % (total_time_estimated - self.v['time_delta_total'], total_time_estimated))

            self.v.update({
                'j': 0,

                'e_j_incomplete': None,
                'e_ji_matrix_incomplete': None,
            })
            self.v['numpy_random_state'] = self.prng.get_state()
            self.save_history()



        self.delete_print_lines(3)
        return self.v, self.history_j

    def post_sample(self):
        for variable in [







                'p_j_cumulative',
                'r_j_cumulative',







                'r_avg_diff_matrix',
                'pr_avg_diff_matrix',


        ]:
            pass












        for variable in [


                'r_avg_diff_matrix',
                'pr_avg_diff_matrix',








        ]:
            self.v[variable + '_normalized'] = h.normalize(self.v[variable])



        for variable in [




        ]:
            self.v[variable] = 1 - self.v[variable]

        t2 = h.now()
        self.print('Finished model generation at', t2)
        self.print('Warmup total time', self.v['time_delta_total_warmup'])
        self.print('Post-warmup total time', self.v['time_delta_total_post_warmup'])
        self.print('Total time', self.v['time_delta_total'])





        self.previous_save_time = h.now() - 2 * self.s.SAVE_TIME_DELTA_MIN
        self.save_history()

    def append_history(self, gene_number, var_name, history):
        dir_path = os.sep.join(['genes'] + [c for c in ('%05d' % gene_number)])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.sep.join([dir_path, var_name + '.h5'])
        with tables.open_file(file_path, mode='a') as f:
            if not hasattr(f.root, 'array'):
                f.create_earray(f.root, 'array', obj=history)
            else:






                f.root.array.append(history)

    def should_save_history(self):
        if self.s.SAVE_TRACE:
            if self.v['iteration'] >= self.s.WARMUP:
                return True


            return self.s.SAVE_WARMUP


        return False

    @h.prevent_sigint
    def save_history(self):
        t1 = h.now()
        if not self.should_save_history():









            if (t1 - self.previous_save_time) > self.s.SAVE_TIME_DELTA_MIN:
                self.print("Started saving results to disk, please don't stop the program now...", t1)
                self.print(" * Backing up existing files...")
                if os.path.isfile(self.v_fname) and os.stat(self.v_fname).st_size != 0:
                    shutil.copyfile(self.v_fname, self.v_fname + '~')

                with open(self.v_fname, 'wb') as f_variables:
                    self.print(' * Saving (only) current iteration data...')
                    self.v['numpy_random_state'] = self.prng.random.get_state()
                    pickle.dump(self.v, f_variables)
                self.previous_save_time = h.now()

                self.print('Saving finished at %s. It took' % self.previous_save_time, self.previous_save_time - t1)
                return






        if self.should_save_history() and ((t1 - self.previous_save_time) > self.s.SAVE_TIME_DELTA_MIN):
            self.print("Started saving results to disk, please don't stop the program now...", t1)
            self.print(" * Backing up existing files...")
            for _fname in (self.v_fname,):
                if os.path.isfile(_fname) and os.stat(_fname).st_size != 0:
                    shutil.copyfile(_fname, _fname + '~')


            with open(self.v_fname, 'wb') as f_variables:
                self.print(' * Saving current iteration data...')
                self.v['numpy_random_state'] = self.prng.get_state()
                pickle.dump(self.v, f_variables)

            history_j = {}
            lengths = set()
            for key in self.history_j:
                if self.history_j[key]:








                    history_j[key] = np.stack(self.history_j[key])
                    lengths.add(len(history_j[key]))

            if (len(lengths) == 1) and (next(iter(lengths)) > 0):
                self.print(' * Saving sampling history data (%d iterations since last save)...' % (next(iter(lengths))))
                for i in range(self.first_gene, self.first_gene + self.J):
                    for key in self.history_j:








                        self.append_history(i, key, history_j[key][:, i - self.first_gene])
            elif len(lengths) == 0:


                pass
            else:
                print(lengths)
                raise Exception("This shouldn't happen")

            for key in history_j:
                self.history_j[key] = []
            self.previous_save_time = h.now()
            self.print('Saving finished at %s. It took' % self.previous_save_time, self.previous_save_time - t1, end='.')
            if len(lengths) != 0:
                _length = next(iter(lengths))
                self.print(' %s per iteration.' % ((self.previous_save_time - t1)/_length))
            else:
                self.print()

    def sample_pymc3(self):
        import pymc3 as pm

        with pm.Model() as model:
            p_j_list = []
            for _j in range(self.J):


                p_j_list.append(pm.Beta('p_%d' % _j, alpha=self.s.A0, beta=self.s.B0))
                r_i_list = []
                for _i in range(self.N):


                    r_i_list.append(pm.Gamma('r_%d_%d' % (_j, _i), self.s.E0, 1.0 / self.s.F0))

































