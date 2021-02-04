import os
import pickle

import numpy as np
import tables
import statsmodels.stats.multitest

import tensorflow as tf

try:
    import tensorflow_probability

    tfpd = tensorflow_probability.distributions
except ModuleNotFoundError:


    tfpd = tf.distributions

from outpyr import helpers as h


def F64(x): return tf.cast(x, dtype=tf.float64)


def skipped(f):
    def _f(self, *args, **kwargs):
        return f(self, *args, **kwargs)[self.skip:]
    return _f


class TraceInspector:
    def __init__(self, trace_data_folder, data=None):
        self.trace_data_folder = trace_data_folder

        if data is None:
            self.data = np.load(os.path.join(trace_data_folder, 'data.npy'))
        else:
            self.data = data

        self.df = h.csv_to_df(os.path.join(trace_data_folder, 'data_frame.csv'))

        with open(os.path.join(self.trace_data_folder, 'variables.pickle'), 'rb') as f:
            self.v = pickle.load(f)

        if not self.v['SAVE_TRACE']:
            raise Exception('The trace was not saved for this model.')

        if self.v['SAVE_WARMUP']:


            self.skip = 1 + self.v['WARMUP']
        else:
            self.skip = 0

    def get_b_j(self, j): return np.mean(self.get_b_j_trace(j))

    def get_e_j(self, j): return np.mean(self.get_e_j_trace(j))

    def get_f_j(self, j): return np.mean(self.get_f_j_trace(j))

    def get_p_j(self, j): return np.mean(self.get_p_j_trace(j))

    def get_r_j(self, j): return np.mean(self.get_r_j_trace(j))

    def get_mu_j(self, j): return np.mean(self.get_mu_j_trace(j))

    def get_var_j(self, j): return np.mean(self.get_var_j_trace(j))

    def get_si_j_i(self, j, i): return h.nb_si(self.data[j, i], self.get_p_j(j), self.get_r_j(j))

    def get_log10si_j_i(self, j, i): return h.nb_log10si(self.data[j, i], self.get_p_j(j), self.get_r_j(j))

    def get_si_j_row(self, j):
        return np.array([self.get_si_j_i(j, i) for i in range(self.data.shape[1])])

    def get_log10si_j_row(self, j): return np.array([self.get_log10si_j_i(j, i) for i in range(self.data.shape[1])])

    def get_si_matrix(self):
        si_matrix = np.zeros((self.data.shape[0], self.data.shape[1]))
        for j in range(self.data.shape[0]):
            si_matrix[j] = self.get_si_j_row(j)
        return si_matrix

    def get_log10si_matrix(self):
        log10si_matrix = np.zeros((self.data.shape[0], self.data.shape[1]))
        for j in range(self.data.shape[0]):
            log10si_matrix[j] = self.get_log10si_j_row(j)
        return log10si_matrix

    @skipped
    def get_b_j_trace(self, j): return self._get_param_j_trace('p', j)

    @skipped
    def get_e_j_trace(self, j): return self._get_param_j_trace('p', j)

    @skipped
    def get_f_j_trace(self, j): return self._get_param_j_trace('p', j)

    @skipped
    def get_p_j_trace(self, j): return self._get_param_j_trace('p', j)

    @skipped
    def get_r_j_trace(self, j): return self._get_param_j_trace('r', j)

    @skipped
    def get_mu_j_trace(self, j):
        p_j_trace = self._get_param_j_trace('p', j)
        r_j_trace = self._get_param_j_trace('r', j)
        return h.get_mu(p_j_trace, r_j_trace)

    @skipped
    def get_var_j_trace(self, j):
        p_j_trace = self._get_param_j_trace('p', j)
        r_j_trace = self._get_param_j_trace('r', j)
        return h.get_var(p_j_trace, r_j_trace)

    @skipped
    def get_si_j_i_trace(self, j, i):
        si_trace = []

        p_j_trace = self._get_param_j_trace('p', j)
        r_j_trace = self._get_param_j_trace('r', j)
        for iteration, p in enumerate(p_j_trace):
            r = r_j_trace[iteration]
            x = self.data[j, i]

            si_trace.append(h.nb_si(x, p, r))
        return np.array(si_trace)

    @skipped
    def get_log10si_j_i_trace(self, j, i):
        p_j_trace = self._get_param_j_trace('p', j)
        r_j_trace = self._get_param_j_trace('r', j)

        log10si_trace = np.zeros_like(p_j_trace)

        for iteration, p in enumerate(p_j_trace):
            r = r_j_trace[iteration]
            x = self.data[j, i]

            log10si_trace[iteration] = h.nb_log10si(x, p, r)
        return log10si_trace

    def plot_b_j_trace(self, j): h.plot(self.get_p_j_trace(j))

    def plot_e_j_trace(self, j): h.plot(self.get_p_j_trace(j))

    def plot_f_j_trace(self, j): h.plot(self.get_p_j_trace(j))

    def plot_p_j_trace(self, j): h.plot(self.get_p_j_trace(j))

    def plot_r_j_trace(self, j): h.plot(self.get_r_j_trace(j))

    def plot_mu_j_trace(self, j): h.plot(self.get_mu_j_trace(j))

    def plot_var_j_trace(self, j): h.plot(self.get_var_j_trace(j))

    def plot_si_j_i_trace(self, j, i): h.plot(self.get_si_j_i_trace(j, i))

    def plot_log10si_j_i_trace(self, j, i): h.plot(self.get_log10si_j_i_trace(j, i))

    @skipped
    def get_param_j_trace(self, param, j):
        return self._get_param_j_trace(param, j)

    @skipped
    def get_param_i_trace(self, param, i):
        return self._get_param_i_trace(param, i)

    def _get_param_j_trace(self, param, j):
        with tables.open_file(os.sep.join([self.trace_data_folder, 'genes'] + [c for c in ('%05d' % j)] + [param + '.h5']), 'r') as h5:
            array = np.array(h5.root.array)
        return array.ravel()

    def _get_param_i_trace(self, param, i):
        with tables.open_file(os.sep.join([self.trace_data_folder, 'samples'] + [c for c in ('%05d' % i)] + [param + '.h5']), 'r') as h5:
            array = np.array(h5.root.array)
        return array.ravel()

    def _get_param_ji_trace(self, param, j, i):
        with tables.open_file(os.sep.join([self.trace_data_folder, 'genes'] + [c for c in ('%05d' % j)] + ['samples'] + [c for c in ('%05d' % i)] + [param + '.h5']), 'r') as h5:
            array = np.array(h5.root.array)
        return array

    def _get_cdf_ji_trace(self, j, i):
        p_j_array = self._get_param_j_trace('p', j)
        p_j_array.shape = (p_j_array.size, 1)
        r_j_array = self._get_param_j_trace('r', j)
        r_j_array.shape = (r_j_array.size, 1)
        P = F64(np.tile(p_j_array, self.data.shape[1]))
        R = F64(np.tile(r_j_array, self.data.shape[1]))
        nb = tfpd.NegativeBinomial(total_count=R, probs=P);

        C = F64(np.tile(self.data[j], (self.v['iteration'] + 1, 1)))
        cdf_ = nb.cdf(C).numpy()

        return cdf_[:, i]

    @skipped
    def get_cdf_ji_trace(self, j, i): return self._get_cdf_ji_trace(j, i)

    def _get_p_value_ji_trace(self, j, i):
        cdf = self._get_cdf_ji_trace(j, i)
        return 2 * np.vstack([cdf, 1 - cdf]).min(axis=0)

    @skipped
    def get_p_value_ji_trace(self, j, i): return self._get_p_value_ji_trace(j, i)

    def set_final_values_from_trace(self, genes=None):
        print('Warning: skipping first %d iterations' % self.skip)
        if genes is None:
            genes = range(self.data.shape[0])



            self.v['p_j_final'] = np.zeros((self.data.shape[0], 1))
            self.v['r_j_final'] = np.zeros((self.data.shape[0], 1))
            self.v['mu_j_final'] = np.zeros((self.data.shape[0], 1))
            self.v['var_j_final'] = np.zeros((self.data.shape[0], 1))









            self.v['p_j_cumulative'] = np.zeros((self.data.shape[0], 1))
            self.v['r_j_cumulative'] = np.zeros((self.data.shape[0], 1))
            self.v['mu_j_cumulative'] = np.zeros((self.data.shape[0], 1))
            self.v['var_j_cumulative'] = np.zeros((self.data.shape[0], 1))

            self.v['p_values_mean'] = np.zeros(self.data.shape)
            self.v['p_values_mode'] = np.zeros(self.data.shape)
            self.v['p_values_var'] = np.zeros(self.data.shape)



        for _iteration, j in enumerate(genes):
            if j != 0:
                h.delete_print_lines(1)
            print('%s: Processing trace directory %s gene %d' % (h.now(), self.trace_data_folder, j))
            p_j_array = self.get_p_j_trace(j)
            p_j_array.shape = (p_j_array.size, 1)
            r_j_array = self.get_r_j_trace(j)
            r_j_array.shape = (r_j_array.size, 1)
            mu_j_array = h.get_mu(p_j_array, r_j_array)
            var_j_array = h.get_var(p_j_array, r_j_array)

            self.v['p_j_final'][j, 0] = p_j_array.mean()
            self.v['r_j_final'][j, 0] = r_j_array.mean()
            self.v['mu_j_final'][j, 0] = mu_j_array.mean()
            self.v['var_j_final'][j, 0] = var_j_array.mean()

            self.v['p_j_cumulative'][j, 0] = p_j_array.sum()
            self.v['r_j_cumulative'][j, 0] = r_j_array.sum()
            self.v['mu_j_cumulative'][j, 0] = mu_j_array.sum()
            self.v['var_j_cumulative'][j, 0] = var_j_array.sum()

            P = F64(np.tile(p_j_array, self.data.shape[1]))
            R = F64(np.tile(r_j_array, self.data.shape[1]))
            nb = tfpd.NegativeBinomial(total_count=R, probs=P);

            C = F64(np.tile(self.data[j], (self.v['iteration'] + 1 - self.skip, 1)))
            cdf_ = nb.cdf(C).numpy()
            self.v['p_values_mean'][j] = cdf_.mean(axis=0)
            self.v['p_values_var'][j] = cdf_.var(axis=0)
            for i in range(self.data.shape[1]):
                self.v['p_values_mode'][j][i] = h.mode_of_continuous(cdf_[:, i], 10)



        print('Saving...')
        with open(os.path.join(self.trace_data_folder, 'variables.pickle'), 'wb') as f_variables:
            pickle.dump(self.v, f_variables)
        print('Finished!')

    def get_p_value_matrix(self):
        print('Calculating the p-value matrix...')
        return 2 * np.minimum(self.v['p_values_mean'], 1 - self.v['p_values_mean'])



    def get_p_value_matrix_old(self):
        fname = os.path.join(self.trace_data_folder, 'p-values.pickle')
        if os.path.isfile(fname):


            with open(fname, 'rb') as f_p_values:
                return pickle.load(f_p_values)
        else:


            print('Calculating the p-value matrix...')


            import tensorflow as tf
            try:
                import tensorflow_probability
                tfpd = tensorflow_probability.distributions
            except ModuleNotFoundError:


                tfpd = tf.distributions

            def F64(x): return tf.cast(x, dtype=tf.float64)

            C = F64(self.data)


            P = F64(np.tile(self.v['p_j_final'], self.data.shape[1]))
            R = F64(np.tile(self.v['r_j_final'], self.data.shape[1]))
            nb = tfpd.NegativeBinomial(total_count=R, probs=P)
            p_values__ = 2 * np.array([1/2*np.ones_like(C), nb.cdf(C).numpy(), 1 - nb.cdf(C).numpy()]).min(axis=0)








            return p_values__


        raise Exception("This shouldn't happen.")

    def get_p_value_sample_adjusted_matrix(self):
        fname = os.path.join(self.trace_data_folder, 'p-values-adjusted.pickle')
        if os.path.isfile(fname):


            with open(fname, 'rb') as f_p_values:
                return pickle.load(f_p_values)
        else:


            print('Calculating the p-value adjusted matrix...')
            p_values__ = self.get_p_value_matrix()
            p_values_adjusted__ = np.empty_like(p_values__)
            for i in range(p_values__.shape[1]):
                p_values_adjusted__[:, i] = statsmodels.stats.multitest.multipletests(p_values__[:, i], method="fdr_by")[1]








            return p_values_adjusted__


        raise Exception("This shouldn't happen.")

    def get_p_value_gene_adjusted_matrix(self):
        fname = os.path.join(self.trace_data_folder, 'p-values-adjusted.pickle')
        if os.path.isfile(fname):


            with open(fname, 'rb') as f_p_values:
                return pickle.load(f_p_values)
        else:


            print('Calculating the p-value adjusted matrix...')
            p_values__ = self.get_p_value_matrix()
            p_values_adjusted__ = np.empty_like(p_values__)
            for j in range(p_values__.shape[0]):
                p_values_adjusted__[j, :] = statsmodels.stats.multitest.multipletests(p_values__[j, :], method="fdr_by")[1]








            return p_values_adjusted__


        raise Exception("This shouldn't happen.")

    def get_z_score_matrix(self):
        p__ = np.tile(self.v['p_j_final'], self.data.shape[1])
        r__ = np.tile(self.v['r_j_final'], self.data.shape[1])
        mu__ = h.get_mu(p__, r__)
        l_ji__ = np.log2((self.data + 1) / (mu__ + 1))
        l_j_ = l_ji__.mean(axis=1)
        l_j_.shape = (self.data.shape[0], 1)
        l_j_std_ = l_ji__.std(axis=1)
        l_j_std_.shape = (self.data.shape[0], 1)
        return (l_ji__ - l_j_) / l_j_std_


t1 = h.now()
C = 0
def get_p_value(c, p, r):
    global C, t1
    print("Processing count %d/%d" % (C, 10556*119))
    print("Time now", h.now())
    print("Time since beginning", h.now() - t1)
    C += 1

    print(c, p, r)

    return tfpd.NegativeBinomial(total_count=F64(r), probs=F64(p)).cdf(F64(c)).numpy()
