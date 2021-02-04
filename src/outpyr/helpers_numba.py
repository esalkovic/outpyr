import numba
import numpy as np
import scipy.special

import outpyr.helpers as h



ENABLE_JIT = True
NOPYTHON = True


def _numba_jit(*args1, **kwargs1):
    if ENABLE_JIT:
        return numba.jit(*args1, **kwargs1)
    else:
        def _decorator(_f):
            return _f
        return _decorator


@_numba_jit(nopython=NOPYTHON)
def column_to_matrix(column, number_of_columns):
    assert len(column.shape) == 2
    assert column.shape[1] == 1

    matrix = np.zeros((column.shape[0], number_of_columns))
    for i in range(number_of_columns):
        for j in range(column.shape[0]):
            matrix[j, i] = column[j, 0]
    return matrix


@_numba_jit(nopython=NOPYTHON)
def row_to_matrix(row, number_of_rows):
    assert len(row.shape) == 2
    assert row.shape[0] == 1

    matrix = np.zeros((number_of_rows, row.shape[1]))
    for j in range(number_of_rows):
        for i in range(row.shape[1]):
            matrix[j, i] = row[0, i]
    return matrix


def sync_prng_state(prng):
    def _decorator(f):
        def f_new(*args, **kwargs):
            integers, index = prng.get_state()[1:3]
            state_pointer = numba._helperlib.rnd_get_np_state_ptr()
            numba._helperlib.rnd_set_state(state_pointer, (index, [int(x) for x in integers]))

            result = f(*args, **kwargs)

            state_pointer = numba._helperlib.rnd_get_np_state_ptr()
            index, integers = numba._helperlib.rnd_get_state(state_pointer)
            prng.set_state(('MT19937', integers, index, 0, 0.0))

            return result
        return f_new
    return _decorator


@_numba_jit(nopython=NOPYTHON)
def crt_p_j_r_j(data, J, N, r_j, E0, e_j_array, e_ji_matrix):
    for _j in range(0, J):
        _r_j_previous = r_j[_j]



        _l_i_array = np.zeros(N, dtype=h.COUNT_INT)



        size = np.float64(np.max(data[_j]))
        bernoulli_p_m_array = np.arange(size)
        bernoulli_p_m_array += _r_j_previous
        np.reciprocal(bernoulli_p_m_array, bernoulli_p_m_array)
        bernoulli_p_m_array *= _r_j_previous



        for _i in range(N):
            _bernoulli_final = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _k in range(data[_j, _i]):


                _bernoulli_final[_k] = np.random.binomial(1, bernoulli_p_m_array[_k])
            _l_i_array[_i] = _bernoulli_final.sum()
            e_ji_matrix[_j][_i] = E0 + N * _l_i_array[_i]
        e_j_array[_j] = E0 + _l_i_array.sum()
    return e_j_array, e_ji_matrix


@_numba_jit(nopython=NOPYTHON)
def crt_p_j_r_i(data, J, N, r_i_, E0, e_i_, e_ji__):
    for _i in range(N):
        _r_i_previous = r_i_[0, _i]



        _l_j_ = np.zeros(J, dtype=h.COUNT_INT)



        size = np.float64(np.max(data[:, _i]))
        bernoulli_p_m_ = np.arange(size)
        bernoulli_p_m_ += _r_i_previous
        np.reciprocal(bernoulli_p_m_, bernoulli_p_m_)
        bernoulli_p_m_ *= _r_i_previous



        for _j in range(J):
            _bernoulli_ = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _k in range(data[_j, _i]):


                _bernoulli_[_k] = np.random.binomial(1, bernoulli_p_m_[_k])
            _l_j_[_j] = _bernoulli_.sum()
            e_ji__[_j][_i] = E0 + N * _l_j_[_j]
        e_i_[0, _i] = E0 + _l_j_.sum()
    return e_i_, e_ji__


@_numba_jit(nopython=NOPYTHON)
def crt_p_i_r_j(data, J, N, r_j_, E0, e_j_, e_ji__):
    for _j in range(J):
        _r_j_previous = r_j_[_j]



        _l_i_ = np.zeros(N)



        size = np.float64(np.max(data[_j, :]))
        bernoulli_p_m_ = np.arange(size)
        bernoulli_p_m_ += _r_j_previous
        np.reciprocal(bernoulli_p_m_, bernoulli_p_m_)
        bernoulli_p_m_ *= _r_j_previous



        for _i in range(N):


            _bernoulli_ = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _k in range(data[_j, _i]):


                _bernoulli_[_k] = np.random.binomial(1, bernoulli_p_m_[_k])
            _l_i_[_i] = _bernoulli_.sum()
            e_ji__[_j][_i] = E0 + N * _l_i_[_i]

        e_j_[_j, 0] = E0 + _l_i_.sum()
    return e_j_, e_ji__


@_numba_jit(nopython=NOPYTHON)
def crt_p_ji_r_j(data, J, N, r_j_, E0, e_j_, e_ji__):
    for _j in range(J):
        _r_j = r_j_[_j]



        _l_i_ = np.zeros(N)



        bernoulli_p_m_ = np.arange(np.float64(np.max(data[_j])))
        bernoulli_p_m_ += _r_j
        np.reciprocal(bernoulli_p_m_, bernoulli_p_m_)
        bernoulli_p_m_ *= _r_j



        for _i in range(N):
            _bernoulli_ = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _m in range(data[_j, _i]):


                _bernoulli_[_m] = np.random.binomial(1, bernoulli_p_m_[_m])
            _l_i_[_i] = _bernoulli_.sum()
            e_ji__[_j, _i] = E0 + N * _l_i_[_i]

        e_j_[_j] = E0 + _l_i_.sum()


    return e_j_, e_ji__


@_numba_jit(nopython=NOPYTHON)
def crt_p_j_r_ji(data, J, N, r_ji__, E0, e_ji__):
    for _j in range(J):
        for _i in range(N):
            _bernoulli_ = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _m in range(data[_j, _i]):


                _bernoulli_[_m] = np.random.binomial(1, r_ji__[_j, _i]/(_m + r_ji__[_j, _i]))
            e_ji__[_j, _i] = E0 + _bernoulli_.sum()
    return e_ji__


@_numba_jit(nopython=NOPYTHON)
def crt_p_ji_r_i(data, J, N, r_i_, E0, e_i_, e_ji__):
    for _i in range(N):
        _r_i = r_i_[0, _i]



        _l_j_ = np.zeros(J)



        bernoulli_p_m_ = np.arange(np.float64(np.max(data[:, _i])))
        bernoulli_p_m_ += _r_i
        np.reciprocal(bernoulli_p_m_, bernoulli_p_m_)
        bernoulli_p_m_ *= _r_i



        for _j in range(J):
            _bernoulli_ = np.empty(data[_j, _i], dtype=h.COUNT_INT)
            for _m in range(data[_j, _i]):


                _bernoulli_[_m] = np.random.binomial(1, bernoulli_p_m_[_m])
            _l_j_[_j] = _bernoulli_.sum()
            e_ji__[_j, _i] = E0 + J * _l_j_[_j]

        e_i_[0, _i] = E0 + _l_j_.sum()


    return e_i_, e_ji__


@_numba_jit(parallel=True, forceobj=True)
def get_p_value_matrix(c__, p__, r__):
    assert c__.shape == p__.shape == r__.shape
    result = np.zeros_like(c__)
    t1 = h.now()
    print("Started at", t1)
    for j in range(c__.shape[0]):
        print(j)
        t_j = h.now()
        print(t_j)
        print("Total time since beginning", t_j - t1)
        for i in range(c__.shape[1]):
            result[j, i] = h.get_p_value(c__[j, i], p__[j, i], r__[j, i])
    t2 = h.now()
    print("Finished at", t2)
    print("Total time it took", t2 - t1)
    return result
