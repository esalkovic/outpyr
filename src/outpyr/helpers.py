import shutil
import subprocess
import sys
import os
import time
from io import StringIO
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

import multiprocess
import signal
import pickle
import pprint
import datetime
import types
from collections import namedtuple
import tempfile

import numpy as np
import pandas as pd
import scipy
import scipy.special
import scipy.stats


from matplotlib import figure
import matplotlib.backends.backend_tkagg as backend
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mpmath as mp
from scipy.stats import nbinom

LOW_RANDINT = 0


HIGH_RANDINT = 2 ** 31 - 1



COUNT_INT = np.uint64
if COUNT_INT == np.uint32:


    HIGHEST_COUNT = np.uint32(2 ** 24 - 1)
elif COUNT_INT == np.uint64:




    HIGHEST_COUNT = np.uint64(2 ** 24 - 1)

LOWEST_COUNT = 0.1

SUM_INT = COUNT_INT



KEYBOARD_INTERRUPTED = False

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

BINS = 100


def load_settings(file_name):
    return import_module_from_file(file_name, 'settings')


def import_module_from_file(file_name, module_name='module'):
    spec = spec_from_loader(module_name, SourceFileLoader(module_name, file_name))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def open_with_default_app(file_name):
    return_code = 0

    if sys.platform.startswith('linux'):
        return_code = subprocess.call(['xdg-open', file_name])

    elif sys.platform.startswith('darwin'):
        return_code = subprocess.call(['open', file_name])

    elif sys.platform.startswith('win'):
        return_code = subprocess.call(['start', file_name], shell=True)

    return return_code


class plt:
    @classmethod
    def plot(cls, *args, **kwargs):
        fig = figure.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(*args, **kwargs)
















        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name)


        open_with_default_app(tmp.name)

    @classmethod
    def scatter(cls, *args, **kwargs):
        fig = figure.Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.scatter(*args, **kwargs)

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name)


        open_with_default_app(tmp.name)

    @classmethod
    def hist(cls, *args, **kwargs):
        fig = figure.Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.hist(*args, **kwargs)

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name)


        open_with_default_app(tmp.name)


def mode(a):
    """
    https://stackoverflow.com/questions/46365859/what-is-the-fastest-way-to-get-the-mode-of-a-numpy-array
    """
    if len(a.shape) != 1:


        if a.shape.count(1) >= (len(a.shape) - 1):
            a = np.ravel(a)
        else:
            raise Exception('Input array should be one-dimensional')
    values, counts = np.unique(a, return_counts=True)
    m = counts.argmax()
    return values[m]


def mode_of_nb_mean_trace(a):
    if len(a.shape) != 1:


        if a.shape.count(1) >= (len(a.shape) - 1):
            a = np.ravel(a)
        else:
            raise Exception('Input array should be one-dimensional')

    limit_range = a.max() - a.min()
    while True:
        bins = COUNT_INT(np.round(np.sqrt(len(a))))
        if bins <= 10:
            return COUNT_INT(np.round(np.median(a)))
        y, x = np.histogram(a, bins=bins)

        i = y.argmax()







        a = a[a > x[i]]
        a = a[a < x[i+1]]

        limit_range = x[i+1] - x[i]


@np.vectorize
def _mode_of_nb_mathematical(p, r):



    if r <= 1:
        return 0
    else:
        return COUNT_INT(np.floor(p * (r - 1)/(1 - p)))


def mode_of_nb_mathematical(p, r):
    mode_ = _mode_of_nb_mathematical(p, r)
    if np.size(mode_) == 1:
        return COUNT_INT(mode_)


def mode_of_continuous(a, bins=BINS):
    if len(a.shape) != 1:


        if a.shape.count(1) >= (len(a.shape) - 1):
            a = np.ravel(a)
        else:
            raise Exception('Input array should be one-dimensional')



    while True:






        if len(a) == 0:
            break
        _bins = COUNT_INT(np.round(np.sqrt(len(a))))
        if _bins <= 20:
            return np.median(a)

        y, x = np.histogram(a, bins=bins)

        i = y.argmax()

        a = a[a > x[i - 1]]
        a = a[a < x[i + 1]]

        if len(set(a)) == 1:
            return a[0]
        elif len(a) == 0:
            return x[i]
    raise Exception("This shouldn't happen.")


def mode_scipy(iterable):
    return scipy.stats.mode(iterable).mode[0]


def delete_print_lines(n):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


mp_power = np.frompyfunc(mp.power, 2, 1)
mp_binomial = np.frompyfunc(mp.binomial, 2, 1)
mp_log = np.frompyfunc(mp.log, 1, 1)
mp_fabs = np.frompyfunc(mp.fabs, 1, 1)
mp_fdiv = np.frompyfunc(mp.fdiv, 2, 1)
mp_hyp2f1 = np.frompyfunc(mp.hyp2f1, 4, 1)
mp_fsub = np.frompyfunc(mp.fsub, 2, 1)


def mp_fprod2(a, b):
    return mp.fprod([a, b])


def mp_fprod(list_):
    f = np.frompyfunc(mp_fprod2, 2, 1)
    res = mp.mpf('1')
    for e in list_:
        res = f(res, e)
    return res


def mp_fsum2(a, b):
    return mp.fsum([a, b])


def mp_fsum(list_):
    f = np.frompyfunc(mp_fsum2, 2, 1)
    res = mp.mpf('0')
    for e in list_:
        res = f(res, e)
    return res


def _mp_round(n):
    ceil = mp.ceil(n)
    floor = mp.floor(n)
    ceil_diff = mp.fabs(n - ceil)
    floor_diff = mp.fabs(n - floor)
    if ceil_diff <= floor_diff:
        return ceil
    else:
        return floor


mp_round = np.frompyfunc(_mp_round, 1, 1)


def _mp_int(n):
    if n > HIGHEST_COUNT:
        n = HIGHEST_COUNT
    return COUNT_INT(n)


mp_int = np.frompyfunc(_mp_int, 1, 1)


def mp_gmean(array):
    return mp_power(mp_fprod((mp.mpf(str(e)) for e in array)), (1.0 / len(array)))


def _mp_to_np_COUNT_INT(n):
    n = mp.mpf(n)
    if n > HIGHEST_COUNT:
        return HIGHEST_COUNT
    else:
        return COUNT_INT(n)


def mp_to_np_COUNT_INT(array):
    return np.frompyfunc(_mp_to_np_COUNT_INT, 1, 1)(array).astype(COUNT_INT)


def convert_settings_module_to_dict(settings):
    result = {}
    for key in dir(settings):
        if not key.startswith('_'):
            value = getattr(settings, key)
            if not isinstance(value, (types.ModuleType, types.FunctionType)):
                result[key] = value


    return result




def gini(array):
    count = array.size
    coefficient = 2 / count
    indexes = np.arange(1, count + 1)
    weighted_sum = (indexes * array).sum(dtype=SUM_INT)
    total = array.sum(dtype=SUM_INT)
    constant = (count + 1) / count

    return coefficient * weighted_sum / total - constant


def lorenz(array):




    scaled_prefix_sum = array.cumsum(dtype=SUM_INT) / array.sum(dtype=SUM_INT)



    return np.insert(scaled_prefix_sum, 0, 0)


def parse_slice(v):
    """
    Parses text like python "slice" expression (ie ``-10::2``).

    :param v:
        the slice expression or a lone integer
    :return:
        - None if input is None/empty
        - a ``slice()`` instance (even if input a lone numbrt)
    :raise ValueError:
        input non-empty but invalid syntax
    """
    orig_v = v
    v = v and v.strip()
    if not v:
        return

    try:
        if ':' not in v:


            v = int(v)
            return slice(v, v + 1)

        return slice(*map(lambda x: int(x.strip()) if x.strip() else None,
                          v.split(':')))
    except Exception:
        pass



    raise Exception("Syntax-error in '%s' slice!" % orig_v)


def now():
    return datetime.datetime.now()


def format_time(t):
    return t.strftime('%Y-%m-%d %H:%M:%S')


def parse_timedelta(s):
    try:
        days, hours = s.strip().split(' days, ')
    except ValueError:
        days = '0'
        hours = s.strip()
    t = datetime.datetime.strptime(hours, '%H:%M:%S.%f')
    return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond) + datetime.timedelta(days=int(days))


def myprint(*args, **kwargs):


    print(*args, **kwargs)




def filter_genes(data):
    """Remove genes whose counts are all zeros,
    or have less than one non-zero count on 100 samples."""
    genes_to_drop = []
    for j in data.index:
        zero_counts = [count for count in data.loc[j] if count == 0]
        if len(zero_counts)/len(data.columns) > 0.75:
            genes_to_drop.append(j)
    if genes_to_drop:
        return data.drop(genes_to_drop)
    else:
        return data


def get_size_factors(df):
    _data = df.values
    N = _data.shape[1]
    _rows = []
    for _row in _data:
        if np.all(_row != 0):
            _rows.append(_row)
    J = len(_rows)
    data = np.array(_rows)
    data.shape = (J, N)
    counts_normalized = np.zeros(data.shape, dtype=np.float64)
    for j in range(data.shape[0]):
        _row = np.array(data[j, :])


        counts = np.array([count for count in _row if count != 0])





        if len(counts) > 0:


            denominator = mp_gmean(counts)








        else:
            denominator = 0

        if denominator == 0:
            counts_normalized[j] = np.zeros(data.shape[1], dtype=np.float64)
        else:
            counts_normalized[j] = mp_fdiv(_row, denominator)


    size_factors = np.zeros(data.shape[1], dtype=np.float64)
    for i in range(data.shape[1]):
        column = np.array([count for count in counts_normalized[:, i] if count != 0])
        size_factors[i] = np.median(column)
    return size_factors


def prevent_sigint(f):
    def _sigint_postpone(signal_, frame):
        global KEYBOARD_INTERRUPTED
        if not KEYBOARD_INTERRUPTED:
            print("\nKeyboard interrupt triggered, but will be postponed as this is a critical section of code.")
            KEYBOARD_INTERRUPTED = True

    def _f(*args, **kwargs):
        global KEYBOARD_INTERRUPTED
        signal.signal(signal.SIGINT, _sigint_postpone)
        f(*args, **kwargs)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        if KEYBOARD_INTERRUPTED:
            raise KeyboardInterrupt
            KEYBOARD_INTERRUPTED = False
    return _f


def _plot_mp(f, args, kwargs):




    f(*args, **kwargs)









def plt_non_blocking(f):
    def _f(*args, **kwargs):
        return _plot_mp(f, args, kwargs)
        proc = multiprocess.Process(target=_plot_mp, args=(f, args, kwargs))
        proc.daemon = True
        proc.start()
    return _f


@plt_non_blocking
def plot(*args, **kwargs):plt.plot(*args, **kwargs)


@plt_non_blocking
def scatter(*args, **kwargs):
    if len(args) == 1:
        x = range(len(args[0]))
        args = (x,) + args
    if 's' not in kwargs:
        kwargs['s'] = 5
    plt.scatter(*args, **kwargs)




@plt_non_blocking
def histogram(*args, **kwargs):
    if 'bins' not in kwargs:


        kwargs['bins'] = 100
    plt.hist(*args, **kwargs)




def nb_si(x, p, r):
    return mp_fdiv(mp_fprod([mp_power(1 - 0, -x), mp_power(p, r), mp_hyp2f1(r, r, 1, mp_power(p - 1, 2))]), mp_binomial(x + r - 1, r - 1))




def nb_log10si(x, p, r):
    return -x * np.log10(1 - p) +\
        r * np.log10(p) +\
        np.log10(scipy.special.hyp2f1(r, r, 1, (p - 1)**2)) -\
        np.log10(max(scipy.special.comb(x + r - 1, r - 1), 1))


def get_mu(p, r):


    return r * p / (1 - p)


def get_var(p, r):


    return r * p / (1 - p) ** 2


def likelihood(x, p, r):
    return mp_fprod([mp_binomial(r + x - 1, x), mp_power(p, r), mp_power((1 - p), x)])


def log_comb(n, k):










    return mp_log(mp_binomial(n, k))


def log_likelihood(x, p, r):
    term1 = log_comb(r + x - 1, x)
    term2 = r * mp_log(p)
    term3 = x * mp_log(1 - p)












    return term1 + term2 + term3


def normalize(array):
    if len(array.shape) == 1:
        min_ = array.min()
        max_ = array.max()










    elif len(array.shape) == 2:
        min_ = array.min(axis=1)
        min_.shape = (array.shape[0], 1)
        max_ = array.max(axis=1)
        max_.shape = (array.shape[0], 1)


    else:
        raise Exception('Shape greater than 2 is not supported')
    denominator = (max_ - min_)






    return (array - min_) / denominator










def split_data(data, n_parts):
    """Split the data into n_parts such that
    the final part will be the smallest one"""
    part_length = len(data) // n_parts + 1
    return [data[i:i + part_length] for i in range(0, len(data), part_length)]


def split_data_smart(data, n_parts):
    """Split the data smartly into n_parts
    such that the final part will be
    the smallest one. "Smartly" means
    that it will try to split the data by
    counts, not simply by gene number."""


    counts_total = data.sum(dtype=np.uint64)
    part_length = np.uint64(counts_total // n_parts + 1)
    counts_current_part_running = np.uint64(0)
    parts = []
    low = np.uint64(0)
    for i in range(data.shape[0]):
        counts_current_part_running += data[i, :].sum(dtype=np.uint64)
        if i > 0 and counts_current_part_running >= (part_length * (len(parts) + 1)):
            parts.append(data[low:i, :])
            low = i
    if low <= data.shape[0] - 1:
        parts.append(data[low:, :])
    return parts


def distribute_multiprocessing(data, n_parts, f):
    import multiprocessing

    data_parts = split_data_smart(data, n_parts)

    jobs = []
    for i, data_part in enumerate(data_parts):
        j = multiprocessing.Process(target=f, args=(i, data_part))
        jobs.append(j)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print('Finished')


def split_data_by_max_count_cutoff(data, count_cutoff):


    max_counts = data.max(1)
    counts_above = []
    counts_below = []
    for j in range(len(data.index)):
        if max_counts.iloc[j] <= count_cutoff:
            counts_below.append(j)
        else:
            counts_above.append(j)
    return data.iloc[counts_below], data.iloc[counts_above]


def get_si(x, p, r):
    return (1 - p)**(-x) * p**r * scipy.special.hyp2f1(r, r, 1, (p - 1)**2) / scipy.special.binom(x + r - 1, r - 1)


def get_si_trace(data, hi, j, i):
    p_j = hi.history['p_j']
    r_j = hi.history['r_j']
    return [get_si(data[j, i], historical[j], r_j[iteration][j]) for iteration, historical in enumerate(p_j)]


def save_df_to_csv(data_df, file_name):
    if not os.path.exists(file_name):
        data_df.to_csv(file_name, sep='\t')

        with open(file_name, 'r') as f:


            text = f.read()[1:]

        with open(file_name, 'w') as f:
            f.write(text)
    else:
        print('The file', file_name, 'already exists, not saving...')


def csv_to_df(file_name, dtype=COUNT_INT):
    with open(file_name, 'r') as f:
        text = f.read()

    if text[0] == '\t':


        text = text[1:]

    dtypes = {}
    for column in pd.read_csv(
            StringIO(text),
            sep='\t',
        nrows=0).columns:
        dtypes[column] = dtype

    return pd.read_csv(
        StringIO(text),
        sep='\t',
        dtype=dtypes,
    )


def copy_mtime(src, dst):
    os.utime(dst, (os.path.getatime(dst), os.path.getmtime(src)))


def apply_outrider(data_file, minCounts=True, filterGenes=True):
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    from rpy2 import robjects

    r = robjects.r

    print('Begin R OUTRIDER code')


    importr('OUTRIDER')
    r_table = r['read.table'](data_file)
    ods = r['OutriderDataSet'](countData=r_table)



    ods_filtered_non_expressed = r['filterExpression'](ods, minCounts=minCounts, filterGenes=filterGenes)







    ods_outrided = r['OUTRIDER'](ods_filtered_non_expressed)

    aberrant_matrix = r['aberrant'](ods_outrided)

    print('End R OUTRIDER code')

    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_aberrant_matrix = robjects.conversion.rpy2py(aberrant_matrix)

    return pd_aberrant_matrix


class DataLoader:
    def __init__(self, settings, base_file_name):
        self.settings = settings
        s = settings

        self.base_file_name = base_file_name
        self.data_file_name = self.base_file_name + '.pickle'
        self.data_filtered_file_name = self.base_file_name + '.filtered.pickle'
        self.size_factors_file_name = self.base_file_name + '.size_factors.npy'
        self.data_normalized_sf_file_name = self.base_file_name + '.normalized_sf.npy'
        self.data_corrected_file_name = self.base_file_name + '.corrected.pickle'
        self.z_scores_file_name = self.base_file_name + '.z_scores.pickle'

        self.samples_to_exclude = []

        if os.path.isfile(self.data_file_name):
            self.data = pd.read_pickle(self.data_file_name)
        else:
            _data = csv_to_df(self.base_file_name)

            self.data = pd.DataFrame(
                data=mp_to_np_COUNT_INT(_data.values),
                index=_data.index,
                columns=_data.columns,
            )

            self.data.to_pickle(self.data_file_name)
            copy_mtime(self.base_file_name, self.data_file_name)

        self.J = self.data.shape[0]
        self.N = self.data.shape[1]
        self.process_data()








        if s.GENE_SAMPLE_RANGES[0] == 'ALL':


            gene_range = range(len(self.data.index))
        else:
            gene_range = range(s.GENE_SAMPLE_RANGES[0][0], s.GENE_SAMPLE_RANGES[0][1])

        if s.GENE_SAMPLE_RANGES[1] == 'ALL':


            sample_range = range(len(self.data.columns))
        else:
            sample_range = range(s.GENE_SAMPLE_RANGES[1][0], s.GENE_SAMPLE_RANGES[1][1])

        self.data = self.data.iloc[gene_range, sample_range]

        self.data_with_outliers = None

        self.__data_corrected = None
        self.__z_scores = None



    def process_data(self):
        if os.path.isfile(self.data_filtered_file_name):


            self.data_filtered = pd.read_pickle(self.data_filtered_file_name)
        else:




            self.data_filtered = self.data
            self.data_filtered.to_pickle(self.data_filtered_file_name)
            copy_mtime(self.base_file_name, self.data_filtered_file_name)

        if os.path.isfile(self.size_factors_file_name):
            myprint('Size factors already saved, loading them...')
            self.size_factors = np.load(self.size_factors_file_name)
        else:
            myprint('Calculating size factors...')
            self.size_factors = get_size_factors(self.data_filtered)
            np.save(self.size_factors_file_name, self.size_factors)
            copy_mtime(self.base_file_name, self.size_factors_file_name)
            myprint('Done!')

        self.size_factors_matrix = np.array(
            [self.size_factors] * len(self.data_filtered.index),
            dtype=np.float64,
        )

        self.u_ji = np.log2(self.data.values / self.size_factors_matrix + 1)
        self.u_j_mean = self.u_ji.mean(axis=1)
        self.u_j_sigma = np.sqrt(self.u_ji.var(axis=1))

        if os.path.isfile(self.data_normalized_sf_file_name):


            self.data_normalized_sf = pd.read_pickle(self.data_normalized_sf_file_name)
        else:
            _data_normalized_sf = self.data_filtered.values/self.size_factors

            self.data_normalized_sf = pd.DataFrame(
                data=np.rint(_data_normalized_sf).astype(COUNT_INT),
                columns=self.data_filtered.columns,
                index=self.data_filtered.index
            )
            self.data_normalized_sf.to_pickle(self.data_normalized_sf_file_name)
            copy_mtime(self.base_file_name, self.data_normalized_sf_file_name)

        return self.data_normalized_sf

    @property
    def data_corrected(self):
        if self.__data_corrected is not None:
            return self.__data_corrected


        if os.path.isfile(self.data_corrected_file_name):
            myprint('Corrected data already saved, loading it...')
            self.__data_corrected = pd.read_pickle(self.data_corrected_file_name)
        else:
            try:
                import autoCorrection
            except ModuleNotFoundError:
                myprint('autoCorrection is not installed. Not applying autoCorrection/autoencoder...')
                return None
            else:
                myprint('Applying autoCorrection/autoencoder...')
                self.corrector = autoCorrection.correctors.AECorrector(epochs=self.settings.AUTOCORRECTION_EPOCHS, encoding_dim=self.settings.AUTOCORRECTION_ENCODING_DIM, seed=self.settings.AUTOCORRECTION_SEED)

                indices = []
                columns = []
                for i, column in enumerate(self.data_filtered.columns):
                    if column not in self.samples_to_exclude:
                        columns.append(column)
                        indices.append(i)

                _data_corrected = self.corrector.correct(
                    counts=self.data_filtered[columns].values,
                    size_factors=self.size_factors_matrix[:, indices]
                )
                self.__data_corrected = pd.DataFrame(
                    data=np.rint(_data_corrected).astype(COUNT_INT),
                    columns=columns,
                    index=self.data_filtered.index
                )
                self.__data_corrected.to_pickle(self.data_corrected_file_name)
                copy_mtime(self.base_file_name, self.data_corrected_file_name)
                if os.path.isdir('saved_models'):
                    print("Cleaning up autoCorrection saved_models...")
                    shutil.rmtree('saved_models')
                    print("Done!")
        return self.__data_corrected

    @property
    def z_scores(self):
        if self.__z_scores is not None:
            return self.__z_scores


        if os.path.isfile(self.z_scores_file_name):
            myprint('Z-scores already saved, loading them...')
            self.__z_scores = pd.read_pickle(self.z_scores_file_name)
        elif self.data_corrected is not None:
            log_ji = np.log2((self.data_filtered.values + 1) / (self.data_corrected.values + 1))
            z_scores_values = (log_ji - log_ji.mean(axis=1, keepdims=True)) / log_ji.std(axis=1, keepdims=True)
            self.__z_scores = pd.DataFrame(
                data=z_scores_values,
                index=self.data_corrected.index,
                columns=self.data_corrected.columns,
            )
            self.__z_scores.to_pickle(self.z_scores_file_name)
            copy_mtime(self.base_file_name, self.z_scores_file_name)
        else:
            return None

        return self.__z_scores

    def inject_outlier(self, j, i, z_score, type='over'):
        if self.data_with_outliers is None:
            self.data_with_outliers = self.data.copy()

        if type == 'over':
            exponent_plus = mp_power(2, (self.u_j_mean[j] + np.exp(z_score) * self.u_j_sigma[j]))
            int_plus = mp_int(mp_fprod([self.size_factors[i], exponent_plus]))
            if int_plus <= self.data.values[j, i]:
                raise IndexError("There's already a greater value at the index [%d, %d] you provided." % (j, i))
            self.data_with_outliers.values[j, i] = int_plus
        elif type == 'under':
            exponent_minus = mp_power(2, (self.u_j_mean[j] - np.exp(z_score) * self.u_j_sigma[j]))
            int_minus = mp_int(mp_fprod([self.size_factors[i], exponent_minus]))
            if int_minus >= self.data.values[j, i]:
                raise IndexError("There's already a smaller value at the index [%d, %d] you provided." % (j, i))
            self.data_with_outliers.values[j, i] = int_minus
        else:
            raise ValueError("Unsupported outlier type: " + repr(type))

        return self.data_with_outliers


def inject_outlier(data, size_factors, j, i, z_score, type_='over'):
    data_with_outlier = data.copy()
    size_factors_matrix = np.array(
            [size_factors] * data.shape[0]
        )

    u_ji = np.log2(data / size_factors_matrix + 1)
    u_j_mean = u_ji.mean(axis=1)
    u_j_sigma = np.sqrt(u_ji.var(axis=1))

    if type_ == 'over':
        exponent_plus = mp_power(2, (u_j_mean[j] + np.exp(z_score) * u_j_sigma[j]))
        int_plus = mp_int(mp_fprod([size_factors[i], exponent_plus]))
        if int_plus <= data[j, i]:
            raise IndexError("There's already a greater value at the index you provided.")
        data_with_outlier[j, i] = int_plus
    elif type_ == 'under':
        exponent_minus = mp_power(2, (u_j_mean[j] - np.exp(z_score) * u_j_sigma[j]))
        int_minus = mp_int(mp_fprod([size_factors[i], exponent_minus]))
        if int_minus >= data[j, i]:
            raise IndexError("There's already a smaller value at the index you provided.")
        data_with_outlier[j, i] = int_minus
    else:
        raise ValueError("Unsupported outlier type: " + repr(type_))

    return data_with_outlier


def inject_multiple_outliers_diagonally(data, size_factors, genes, outlier_scores, type_='both'):
    data_with_outliers = data.copy()
    size_factors_matrix = np.array(
            [size_factors] * data.shape[0]
        )

    u_ji = np.log2(data / size_factors_matrix + 1)
    u_j_mean = u_ji.mean(axis=1)
    u_j_sigma = np.sqrt(u_ji.var(axis=1))



    for _j_index, j in enumerate(genes):
        for z, Z in outlier_scores:
            exponent_plus = mp_power(2, (u_j_mean[j] + np.exp(Z) * u_j_sigma[j]))
            exponent_minus = mp_power(2, (u_j_mean[j] - np.exp(Z) * u_j_sigma[j]))

            i_plus = -(2*(z + _j_index*len(outlier_scores)) + 1)
            i_minus = -(2*(z + _j_index*len(outlier_scores)) + 2)
            int_plus = mp_int(mp_fprod([size_factors[i_plus], exponent_plus]))
            int_minus = mp_int(mp_fprod([size_factors[i_minus], exponent_minus]))








            if type_ == 'both':
                insert_plus = True
                insert_minus = True
            elif type_ == 'over':
                insert_plus = True
                insert_minus = False
            elif type_ == 'under':
                insert_plus = False
                insert_minus = True
            else:
                raise Exception("This shouldn't happen.")

            if insert_plus:
                data_with_outliers[j, i_plus] = int_plus
            if insert_minus:
                data_with_outliers[j, i_minus] = int_minus

    return data_with_outliers


def extract_representative_genes(dl, number_of_representative):
    ADD_FROM_BEGINNING = 3
    ADD_FROM_END = 3

    assert number_of_representative > (ADD_FROM_BEGINNING + ADD_FROM_END)

    gene_names = list(dl.data.index)
    mean_gene_list = []
    for i, row in enumerate(dl.data.values):
        mean_gene_list.append((row.mean(), gene_names[i], row))
    mean_gene_list.sort()
    representative_genes = []
    count = 0
    index = []

    for i in range(ADD_FROM_BEGINNING):
        representative_genes.append(mean_gene_list[i][2])
        index.append(mean_gene_list[i][1])
        count += 1

    for i in range(ADD_FROM_BEGINNING , len(mean_gene_list), len(mean_gene_list) // number_of_representative):
        representative_genes.append(mean_gene_list[i][2])
        index.append(mean_gene_list[i][1])
        count += 1
        if count == number_of_representative - ADD_FROM_END:
            break

    for i in range(ADD_FROM_END):
        representative_genes.append(mean_gene_list[-(ADD_FROM_END - i)][2])
        index.append(mean_gene_list[-(ADD_FROM_END - i)][1])

    return pd.DataFrame(
                data=mp_to_np_COUNT_INT(np.array(representative_genes)),
                index=index,
                columns=dl.data.columns,
            )


def nbpmf(c, p, r):
    gammaln = scipy.special.gammaln
    return np.exp(gammaln(c + r) - gammaln(c + 1) - gammaln(r) + c * np.log(p) + r * np.log(1 - p))




def nbpmf_slow(c, p, r):
    return float(mp_fprod([mp.gammaprod([c + r], [c + 1, r]), mp_power(p, c), mp_power((1 - p), r)]))




def nbpmf_imprecise(c, p, r):
    gammaln = scipy.special.gammaln
    return np.exp(gammaln(c + r) - gammaln(c + 1) - gammaln(r) + c * np.log(p) + r * np.log(1 - p))




def nbpmf2(c, p, r):
    return nbinom.pmf(r - 1, c + 1, p) * (1 - p)/p


t1 = now()
C = 0
def get_p_value_old(c, p, r):
    global C, t1
    print("Processing count %d/%d" % (C, 10556*119))
    print("Time now", now())
    print("Time since beginning", now() - t1)
    C += 1

    print(c, p, r)
    if c <= 1:
        res = 2 * min(1 / 2, nbpmf(0, p, r))
    else:


        sum_1 = scipy.stats.beta.cdf(p, r, c + 1)


        sum_ = sum_1 + nbpmf(c, p, r)
        print(sum_)
        print(sum_1)
        print(1 - sum_1)
        res = 2 * min(1/2, sum_, 1 - sum_1)
    if res < 0:
        raise Exception("This shouldn't happen")
        return 0
    else:
        return res


def sort_p_values(pv__):
    pv_sorted_list = []
    for j in range(pv__.shape[0]):
        for i in range(pv__.shape[1]):
            pv_sorted_list.append((pv__[j, i], j, i))
    pv_sorted_list.sort()

    return pv_sorted_list


def convert_settings_module_to_SimpleNamespace(settings):
    settings_dict = {}
    for key in dir(settings):
        if key[0].isalpha() and key[0].isupper():
            settings_dict[key] = getattr(settings, key)
    return types.SimpleNamespace(**settings_dict)


def check_iterable(value):
    try:
        iterator = iter(value)
    except TypeError:
        return [value]
    else:
        return value


def mm2inch(value):
    return value/10/2.54
