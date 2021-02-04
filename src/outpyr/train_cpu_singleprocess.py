

import datetime
import os
import sys
import shutil

import numpy as np

import outpyr.helpers as h
from outpyr.sampler.cpu import p_j_r_j_numba

now = datetime.datetime.now


def run(data_file, settings, preprocessing, gene_subset, model):
    np.random.seed(settings.SEED)
    dl = h.DataLoader(settings, data_file)

    if preprocessing is None:
        df = dl.data
        corrected_string = 'unpreprocessed'
    elif preprocessing == 'autoCorrection':
        df = dl.data_corrected
        if df is None:
            raise ModuleNotFoundError("It seems that autoCorrection is not installed")
        corrected_string = 'autoCorrected'
    elif preprocessing == 'normalize_sf':
        df = dl.data_normalized_sf
        corrected_string = 'normalized_sf'
    else:
        raise Exception("This shouldn't happen")

    data = np.clip(df.values, a_min=None, a_max=h.HIGHEST_COUNT)

    if gene_subset is not None:


        try:
            _index1 = int(gene_subset)
            _index2 = _index1 + 1




            data = data[_index1:_index2]
        except ValueError:
            data = eval('data[%s]' % gene_subset)
    else:
        gene_subset = 'ALL'


















    if model == 'p_j_r_j_numba':
        Sampler = p_j_r_j_numba.PjRjNumbaCPUSampler




























    else:
        raise Exception("This shouldn't happen")





    sampling_id = '[' + gene_subset.replace(':', '_') + ']-sp-' + corrected_string + '-from-' + os.path.splitext(os.path.basename(data_file))[0]
    sampling_dir = os.path.join(os.path.dirname(data_file), sampling_id)

    if os.path.isfile(sampling_dir):
        raise Exception(sampling_dir + 'is an existing file, not a folder!')
    elif not os.path.isdir(sampling_dir):
        os.mkdir(sampling_dir)
        h.save_df_to_csv(df, os.path.join(sampling_dir, 'data_frame.csv'))
        np.save(os.path.join(sampling_dir, 'data.npy'), data)

    os.chdir(sampling_dir)





    t = Sampler(
        sampling_dir,


        settings.SEED,
        settings,






        data,
        0,
        0,
    )

    t.run()

    return sampling_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')
    parser.add_argument('gene_subset', metavar='gene_subset', type=str, nargs='?', help='subset of genes to be used for training')
    parser.add_argument("-s", "--settings", default=None)
    parser.add_argument("-p", "--preprocessing", choices=[None, 'normalize_sf', 'autoCorrection'], default=None)
    parser.add_argument("-m", "--model", dest="model", default="p_j_r_j_numba")
    args = parser.parse_args()

    if args.gene_subset:
        gene_subset_unfiltered = args.gene_subset
        gene_subset = ''
        for c in gene_subset_unfiltered:
            if c in '[]:0123456789-,':
                gene_subset += c
    else:
        gene_subset = None



    data_file = args.data_file[0]


    t1 = now()
    if args.settings is not None:
        settings = h.load_settings(args.settings)
    else:
        import outpyr.settings as settings

        settings.SAVE_TIME_DELTA_MIN = datetime.timedelta(seconds=120)

    if os.path.isfile(data_file):
        run(data_file, settings, args.preprocessing, gene_subset, args.model)
    else:
        parser.error("The file (%s) you provided does not exist." % data_file)
    t2 = now()




if __name__ == '__main__':
    main()
