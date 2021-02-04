

import argparse
import os
import sys

import multiprocessing


import math

import numpy as np

from outpyr import helpers as h


from outpyr.sampler.cpu.p_j_r_j_numba import PjRjNumbaCPUSampler as Sampler







QRNG_HEX_ARRAY = ["f20aa5","0cda68","24bafb","8ecab7","6b4980","847afc","412f65","cae39c","a6f9cb","bf60a5","d268c9","62954e","b05812","4aa437","2df61c","42a6e0","b92ecc","d4e7d0","095642","f9a7f9","87019b","744b94","458b62","fec5ba","203698","15471c","7bd3c2","e713c2","d769c8","29bee0","feb810","6dbe68","fdf0e7","91b73d","e6981c","f99a53","0a3962","062e7b","8aa965","f2cea6","90f242","2d6c9c","09edab","144af5","db587f","1b4eb8","e420c3","0e3d73","69a4a1","69146d","38c78e","5bc7d1","53e6c3","548d1e","cdd643","9601b9","766133","644edc","9f1bc1","7d278e","6b7721","aef27d","6a107c","93fff2","455024","66b0ca","72caf6","dbc873","ab4265","c0767e","a9c699","5b6696","74c797","fe1c42","fe9434","7d1543","c998ff","4360b8","a4d978","77da42","28ef74","9240ab","5272b2","8771a0","6d51b2","14ed77","6510b1","d37078","91f130","051fcb","a4324a","8851fa","1de7a1","707dc7","e9bf84","a1e9de","526c52","63da01","e876bc","296c54","7ae780","a4a97f","89a48d","b5be0a","947ecc","6f8507","2da532","24f0ca","5ce926","ce2efb","dd2e6f","58fbeb","3592b6","89e2fd","0bf1f0","f3530c","6ded2c","ec7902","c1ec4f","cdd738","376e28","8fb8d1","db04f2","c32dad","79918c","e0a68a","7b3985","e14233","02377e","ebec3a","0b4225","97cbee","110ede","55eeb9","346014","02d6da","bc0c17","cb61da","9d9350","dcde55","ba5951","bcbb76","450391","23958b","99cb23","ac84ce","738f74","e9c979","9e1f0f","3073b6","761da4","2342b0","a234ee","e93184","600660","3e15ae","0b360b","85e1d0","66f8a9","c07573","f85e15","54ddcd","4bfc7c","473bfa","687083","57d47d","47d376","328edf","98f788","65bf53","2bde69","a75ec1","882030","c3bd0a","cd756b","3fe33d","b677c0","36373a","71b2ed","00cdc8","731e6d","598546","2a8807","2e0ca7","75ed8d","6faa90","26ca78","df7305","d813e6","2754ec","47440a","6b53a4","8701f8","c5e7fa","571369","69f50c","13353b","a6750c","fd7747","e9792b","a90b90","c9d9ed","bb1c40","84c18a","839072","718046","092fe2","4bcb73","836a92","97ddf0","bc45b3","9296a8","a80b7d","5a5341","b71111","f915a9","a08a76","db1477","f3b9b6","1b38ae","4a7921","7e6a43","2bf42a","6b148f","16d803","a553bf","d72f7f","58a595","4ce3fc","86ce4c","4778bb","437904","f05438","02f396","3d062e","1eef4a","be2317","7711c8","533d69","55f143","7ad1e9","4015ff","0839b3","8f7308","f19b19","592d46","0323eb","3f95e2","77c296","2e7f31","f2bef9","4de67e","ff9e95","556088","87f704","14832a","9703f7","184e21","290a79","91de8e","333171","ad6c1d","6669ea","fc7353","752b28","f34ba9","5befbc","dbb7fd","49a42e","c2958d","118524","646d3b","8f7ce1","1cdd4e","1cdfc6","9ac4e5","048831","09acb6","3ae7a2","511026","459c2c","b74194","c45777","407675","0b5fe3","7fcd8e","2a15ee","16ea78","1a6ce9","bbac69","d1659b","05ef3e","5d9159","d73eff","d06f6b","6880da","32dbcf","8ff581","8481fd","40f228","6ec3c9","b98e03","bbe5b0","1ccade","cf3ce5","4f56bc","76bcd6","17ccc8","b95b46","df4ae1","54dec0","188157","9ac5ab","61c97b","3a99bd","da44b3","49e413","70c6fa","81c6a3","48ffcd","62a9df","56a7be","ddd58c","eebbf9","2cc67a","4f221b","56e053","a4f187","0c5885","f3fc5a","087b36","70f295","063779","5d1859","fd2110","4afe7e","7bacaa","de2546","50d32c","662aee","cd83ed","e1e588","9c6dc3","03986b","48442c","8fb09f","49d986","2791ee","a3dd00","e2bdf4","2c370c","6323f4","1b24b4","55f66c","de679a","f81d84","f98d21","19504f","f83d26","85a104","dae7a1","097bbc","ded2ed","e39bba","4bc954","052460","d5e6e1","3438f2","8412b8","bc4e3b","b3da2d","6f879b","c71403","55bb06","2ee40a","358dd3","76d0de","636ff2","1a07da","bb336b","7c9a8e","af91c4","4277d6","846e3d","5c5826","2caa4d","4175d0","234d78","c32dc0","52e707","f5b883","50f0ee","665188","c412b3","e0142c","a85d52","ccfc9a","7749d9","12caf3","659d18","bf3dd9","2dd72a","2f9b0a","042a64","c31917","34fd0c","7df244","9b4063","45afd7","8d78c7","0a36c0","f66bae","b44793","0d5cc1","5c9a69","98e88e","313c28","8c5cbb","10021b","11a93f","64e8d3","3d6147","d82a7b","ad2876","5557a2","c0e431","e397cd","d82bbc","adc500","50aca4","243391","df76fd","ed645a","31f9a6","657a56","cbfa1f","00d5ef","4983a4","1c2668","20e668","9805d0","c74983","566715","54ee1a","126505","d03dc9","bbc8e7","241473","729af9","8f333c","723f07","4ac3c8","68cdd5","f5c4f7","9e0130","dea8bf","c63a6a","ec4852","2a4946","5d4f10","f141fe","36a5b3","9e42fb","a4b46c","19bd92","e262e6","c610be","3dfcef","002ee2","3d195d","a940c8","b463dd","dbb7c9","fc020e","e70276","058797","3bcc23","ab10a8","28b593","c1b13f","bce841","45bb4f","ca2d2a","db3e2c","c0b638","b06ed7","ead8fb","1e0a46","6bfd24","4b32d3","2defd1","dc56b3","2f4093","49d358","08d45d","e7b469","e03b83","77fd74","199a4f","5c3fdd","0d1b96","861bfa","b56d37","7553df","e37684","00faa1","959ccb","d09168","058773","c0324d","9a4604","4a9388","08a2ae","2171a5","d11a98","4d3a56","d1fd7b","cdf184","2b30b9","f7b6df","4ce034","7c20e7","619588","fd7867","adc8db","27f3a1","3b857c","99160c","f8afad","80e14f","ae3443","a33a98","947186","a8a2bb","c52d7e","e411d2","5b3f85","e8ac2f","dc558f","f02ba2","b95b3a","f44c46","0f19f4","afff3e","89bc43","14ffcd","2c2b52","f853dc","d1b7df","6ca532","411e7f","cdeda4","ac23ab","8a38bd","d8c6bd","0c9fff","41666c","2bc9a6","43520b","28a9c5","7473d1","71c4cb","e241d2","b8a9e3","b16472","f22a4d","911e60","242583","0d2baa","4bd330","9823d3","4f6a03","514c33","98760d","fe9124","e72d9d","baf091","c55938","d81135","9e7ae6","98ed50","fe4cd1","2d1add","8fd300","e3fa1a","25d5ad","20fdf3","cc7837","bb5fed","0b14f5","7cba14","2aa8ad","8012af","95e7d9","05afcc","1bcc42","27cfbc","9a3be4","c16662","526cc0","50e880","26ae41","500878","c0afd2","efab14","124b50","a548a8","99c0c4","d094cd","029b11","a65384","ba29b4","e7583a","931bfc","3c3cbc","f0ad27","689f48","7459cf","d6af06","1d4e05","250552","3ba29b","b85025","2598f1","48bdd1","67bece","2067de","962597","36ae59","181699","d7d175","52a9d5","075e82","d89f7c","cda570","effffc","282fc9","591876","64e7b0","bbb2e4","0aa419","da1f2f","f82884","58e460","34860b","ffbf91","0ca2a8","4e03ce","57c9ad","7646b3","377825","1831c1","615a4d","64a02c","793d6e","a3c2ed","41b8de","645de3","769796","dd4ef8","8318a6","39f111","46b30a","11aa96","104fe3","5d20c9","d37342","1965f8","645e84","5769e1","4dd94e","a1f76e","ade67a","6b46dd","04a6af","e53c0a","f3084a","87a00b","a8f8de","5768d7","ac1f64","9e5c2f","92b947","87bb68","e548aa","3d23db","6cd74a","342b75","db0fc3","654191","adefcd","bb7152","7f7f2b","7a5b44","3fa6d3","f8de2b","33b263","97b8e6","c284b8","5bf439","eac97e","b2c923","03a7c1","ff72bf","b57b53","b10285","759ae1","186c93","505f7e","e0be2b","edda6e","fd0131","5a1ed3","4dd2a6","c35ad1","f8eff8","22c52c","eb58b0","4dbaf5","ef7c16","e43b20","9f49b7","f7a1e3","23de9f","266f02","4d5d31","7233fb","e8646d","6a8971","ca4bfc","e2d433","305124","c8a675","20144b","143d60","7854be","d327eb","077964","400890","4c0e47","62f022","b0c340","477c3f","264da9","76fd2a","bfea48","de324f","8640b3","3bf79b","f69d92","0bbc22","ef8d0d","246c24","13eca3","1576b0","3c7052","20fcde","f68a20","269ce3","b992a7","ea4bb1","e938e8","53d618","cdf7f7","78ef5d","539e62","7fab16","a7ffef","ba153e","6a1f36","fefaef","d03c00","9f5a50","6b1c6c","28d1b9","594687","c26eae","5b9885","33e731","12a902","570849","e9fc09","46d9cc","7ab57f","200c00","337708","4a13f0","b6a3a5","0f36de","d072f6","14d9c2","568660","8734ce","f45136","2795e3","02a3d3","91a11e","c656c0","83af84","cd8a3a","34c6e4","ad4404","711e32","9afd4d","947205","ae75d8","09a8f5","37e0c7","c0b5e4","c1076b","b0f974","4731a1","7a85e3","893c72","ddc7a0","663a4a","7abfce","54320b","6f5268","53df5b","6909b1","bfdc61","99f048","c19e47","f59f7a","3927be","795dea","d46066","148a52","0c10e1","7967b0","581858","4369af","e4472b","2b4da5","5907bd","dca60c","22aea4","c77af4","132ec4","095bba","299937","cf48ed","33c927","8ae0cf","5222d9","e5aa4c","11142c","038057","c28c66","6188a9","203fbc","0c6676","dd5b3b","eebf7a","30dc0d","c2bd11","71ceb8","4c00b7","537932","1ec216","e88c6b","47544c","887a66","7c48e1","4ca7b8","ac1f3b","176870","10150a","de221b","5eb7d6","5b76dd","b4dc0d","73d6d7","9160e7","01c10c","f45ccc","9b7689","9df94a","484ab3","7ee923","7629f9","8f6fc5","533940","f57d7c","b1d579","5fe3e8","657916","035481","9de854","60edfe","b46c26","fd9cc3","f0d3cd","f692d3","a51c15","884964","3fe50f","3e7b7f","785f37","ea75ab","56ef18","0bf524","80b2c6","3b194a","63d35f","fed92a","b878eb","89b8ea","44dba6","098682","ba4530","226306","5548f3","7b3b4e","24b577","33e0aa","0f76bf","6ff756","89e212","d6cf83","19a61e","ac5697","a3f841","33ddea","538aba","ad1fe1","016e79","259886","faa00b","aa4a81","7cfe66","c73f8e","bfe39b","22db2a","deaeba","e93b4b","fea758","385c0b","4fa1c9","e25056","af5aa0","c8cbb6","057db3","d89196","d031c3","7c28b2","f0b771","09f934","72099c","5d9ffb","1c49ac","9a57ad","1eef49","cef7de","093e71","bb0799","8a89fe","e5041d","231f16","83727f","4a768b","ae81e9","26e48f","168bee","a256dd","d88f56","68f164","546cb3","adcf8c","281393","8745fd","52e884","4ff7f8","9dd179","a06c11","b8bcdf","026b01","021c40","999267","8d2a00","a67314","d14e30","3a0bec","6f6eff","5047c1","e1a1c1","37fcf4","4034e1","b87f37","5e84c7","fc23ce","88444f","7f33c2","96b914","fceff2","be45a4","4268db","91714b","b3a880","94395a","fab081","6ba5a9","f7ffed","47078a","2f0386","c0753b","f0aba6","94d96e","987038","4b69fe","34de1d","1e5d8b","cebac8","dddeb2","1fdc72","a30db7","994157","3e7770","eb14fe","7e759b","2fc288","d7214a","19ffaf"]
QRNG_UINT32_ARRAY = [int(_hex, 16) for _hex in QRNG_HEX_ARRAY]


COORDINATOR_SEED = 5786153
WORKERS_SEEDS = QRNG_UINT32_ARRAY
WORKERS = {}




class MySampler(Sampler):
    def save_history(self):
        pass

    def print(self, *args, **kwargs):
        pass

    def delete_print_lines(self, n):
        pass


def f(worker_id, settings, seed, data, first_gene, first_sample, variables):


    if worker_id in WORKERS.keys():
        t = WORKERS[worker_id]
    else:


        t = MySampler('kremer-small-%002d' % worker_id, seed, settings, data, first_gene, first_sample)

        if variables['numpy_random_state'] is None:
            variables['numpy_random_state'] = t.v['numpy_random_state']
        WORKERS[worker_id] = t

    if variables is None:
        raise Exception("This shouldn't happen")
    else:
        t.v = variables
        if t.v['numpy_random_state'] is not None:
            t.prng.set_state(t.v['numpy_random_state'])


    t1 = h.now()



    variables_new, history_j_new = t.sample(settings.ITERATIONS_IN_ONE_BATCH)
    return worker_id, variables_new, history_j_new


def run(data_file, settings, preprocessing, n_parts):
    np.random.seed(settings.SEED)










    scores = [
        'r_avg_diff_matrix',
        'pr_avg_diff_matrix',
    ]

































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

    gene_subset = None
    if gene_subset is not None:


        try:
            _index1 = int(gene_subset)
            _index2 = _index1 + 1




            data = data[_index1:_index2]
        except ValueError:
            data = eval('data[%s]' % gene_subset)
    else:
        gene_subset = 'ALL'

    id_ = '[' + gene_subset.replace(':', '_') + ']-mp-' + corrected_string + '-from-' + os.path.splitext(os.path.basename(data_file))[0]


    folder_name = id_
    if os.path.isfile(folder_name):
        raise Exception(folder_name + 'is an existing file, not a folder!')
    elif not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        h.save_df_to_csv(df, os.path.join(folder_name, 'data_frame.csv'))
        np.save(os.path.join(folder_name, 'data.npy'), data)


    os.chdir(folder_name)





    t = Sampler(id_, COORDINATOR_SEED, settings, data, 0, 0)
    if 'workers_states' not in t.v.keys():
        t.v['workers_states'] = {}


    t.history_j['p'] = []
    t.history_j['r'] = []














    data_parts = h.split_data_smart(data, n_parts)


    if len(data_parts) > (multiprocessing.cpu_count() - 2):
        n_parts = multiprocessing.cpu_count() - 2
        data_parts = h.split_data_smart(data, n_parts)
    if len(data_parts) != n_parts:
        print(len(data_parts), '!=', n_parts)


        raise Exception("Cannot split the data according to the provided number of jobs. Perhaps the total number of genes is small? Consider running the script in singleprocess mode (without the -j/--jobs argument)")







    variables_parts = []
    low = 0
    high = 0


    for i, data_part in enumerate(data_parts):


        high += len(data_part)
        variables = {
            'numpy_random_state': t.v['workers_states'].get(i, None),
            'a_j_array': t.v['a_j_array'][low:high],

            'p_j': t.v['p_j'][low:high],
            'r_j': t.v['r_j'][low:high],





            'p_j_cumulative': t.v['p_j_cumulative'][low:high],
            'r_j_cumulative': t.v['r_j_cumulative'][low:high],





            'am_delta_matrix': t.v['am_delta_matrix'][low:high],
            'log_gm_delta_matrix': t.v['log_gm_delta_matrix'][low:high],
            'd_kl_matrix': t.v['d_kl_matrix'][low:high],

            'am_delta_normalized_matrix': t.v['am_delta_normalized_matrix'][low:high],
            'log_gm_delta_normalized_matrix': t.v['log_gm_delta_normalized_matrix'][low:high],
            'd_kl_normalized_matrix': t.v['d_kl_normalized_matrix'][low:high],

            'time_delta_total': t.v['time_delta_total'],
            'time_delta_total_warmup': t.v['time_delta_total_warmup'],
            'time_delta_total_post_warmup': t.v['time_delta_total_post_warmup'],
            'time_delta_iteration': t.v['time_delta_iteration'],
            'time_delta_gene': t.v['time_delta_gene'],

            'iteration': t.v['iteration'],

            'ITERATIONS': t.v['ITERATIONS'],
            'SAVE_TRACE': t.v['SAVE_TRACE'],
            'WARMUP': t.v['WARMUP'],
            'SAVE_WARMUP': t.v['SAVE_WARMUP'],

            'j': 0,
            'e0_j_incomplete': None,
            'e0_ji_matrix_incomplete': None,
        }
        for key in t.v.keys():
            for score in scores:
                if key.startswith(score):


                    variables[key] = t.v[key][low:high]



        variables_parts.append(variables)


        low = high


    _t1 = h.now()
    if t.v['iteration'] == 0:
        t.previous_save_time = h.now() - 2 * t.s.SAVE_TIME_DELTA_MIN
    else:
        t.previous_save_time = _t1
    while t.v['iteration'] < settings.WARMUP + settings.ITERATIONS:
        h.myprint('=================================')
        if settings.ITERATIONS_IN_ONE_BATCH == 1:
            h.myprint('Starting iteration %d/%d(%d + %d)' % (t.v['iteration'] + 1, settings.WARMUP + settings.ITERATIONS, settings.WARMUP, settings.ITERATIONS))
        elif settings.ITERATIONS_IN_ONE_BATCH > 1:
            h.myprint('Starting %d iterations %d-%d/%d(%d + %d)' % (settings.ITERATIONS_IN_ONE_BATCH, t.v['iteration'] + 1, t.v['iteration'] + settings.ITERATIONS_IN_ONE_BATCH, settings.WARMUP + settings.ITERATIONS, settings.WARMUP, settings.ITERATIONS))

        h.myprint('=================================')
        h.myprint(h.now())
        h.myprint('Using %d virtual cores out of %d' % (n_parts, multiprocessing.cpu_count()))

        settings_list = [h.convert_settings_module_to_SimpleNamespace(settings) for i in range(n_parts)]
        seeds = [WORKERS_SEEDS[i] for i in range(n_parts)]
        first_gene = 0
        first_genes = []
        first_samples = []
        for data_part in data_parts:
            first_genes.append(first_gene)
            first_gene += len(data_part)

            first_samples.append(0)
        with multiprocessing.Pool(processes=n_parts) as pool:
            results = pool.starmap(f, zip(range(n_parts), settings_list, seeds, data_parts, first_genes, first_samples, variables_parts))

        results.sort()



        t.v['workers_states'] = {}

        t.v['a_j_array'] = []

        t.v['p_j'] = []
        t.v['r_j'] = []

        t.v['p_j_cumulative'] = []
        t.v['r_j_cumulative'] = []







        for score in scores:
            for _key in t.v.keys():
                if _key.startswith(score):
                    t.v[_key] = None



        history_j_new = {
            'p': [[] for _i in range(len(results[0][2]['p']))],
            'r': [[] for _i in range(len(results[0][2]['r']))],




        }
        _iteration_parts = []

        variables_parts = []
        time_delta_total_parts = []
        time_delta_iteration_parts = []
        time_delta_gene_parts = []


        for id_, variables_part, history_j_part in results:
            variables_parts.append(variables_part)



            t.v['workers_states'][id_] = variables_part['numpy_random_state']

            t.v['a_j_array'].extend(variables_part['a_j_array'])

            t.v['p_j'].extend(variables_part['p_j'])
            t.v['r_j'].extend(variables_part['r_j'])

            t.v['p_j_cumulative'].extend(variables_part['p_j_cumulative'])
            t.v['r_j_cumulative'].extend(variables_part['r_j_cumulative'])













            for score in scores:
                for _key in variables_part.keys():
                    if _key.startswith(score):
                        if (_key not in t.v) or (t.v[_key] is None):
                            t.v[_key] = variables_part[_key]
                        else:
                            t.v[_key] = np.vstack((t.v[_key], variables_part[_key]))










            if t.should_save_history():
                assert len(history_j_part['p']) == len(history_j_part['r'])
                assert len(history_j_part['p']) == len(history_j_new['p'])
                for iteration in range(len(history_j_part['p'])):
                    history_j_new['p'][iteration].extend(history_j_part['p'][iteration])
                    history_j_new['r'][iteration].extend(history_j_part['r'][iteration])





            time_delta_total_parts.append(variables_part['time_delta_total'])
            time_delta_iteration_parts.append(variables_part['time_delta_iteration'])
            time_delta_gene_parts.append(variables_part['time_delta_gene'])
            t.v['iteration'] = variables_part['iteration']
            _iteration_parts.append(variables_part['iteration'])

        assert len(set(_iteration_parts)) == 1

        t.v['a_j_array'] = np.array(t.v['a_j_array'])

        t.v['p_j'] = np.array(t.v['p_j'])
        t.v['r_j'] = np.array(t.v['r_j'])

        t.v['p_j_cumulative'] = np.array(t.v['p_j_cumulative'])
        t.v['r_j_cumulative'] = np.array(t.v['r_j_cumulative'])







        if t.should_save_history():
            assert len(history_j_new['p']) == len(history_j_new['r'])
            for iteration in range(len(history_j_new['p'])):
                t.history_j['p'].append(np.array(history_j_new['p'][iteration]))
                t.history_j['r'].append(np.array(history_j_new['r'][iteration]))





        _t_iter_delta = h.now() - _t1
        _t1 = h.now()
        t.v['time_delta_total'] += _t_iter_delta
        t.v['time_delta_iteration'] = _t_iter_delta
        t.v['time_delta_gene'] = max(time_delta_gene_parts)

        h.myprint('-------------------------')
        if settings.ITERATIONS_IN_ONE_BATCH == 1:
            h.myprint('Iteration %d took:' % (t.v['iteration']), t.v['time_delta_iteration'])
        elif settings.ITERATIONS_IN_ONE_BATCH > 1:
            h.myprint('%d iterations %d-%d took:' % (settings.ITERATIONS_IN_ONE_BATCH, t.v['iteration'] - settings.ITERATIONS_IN_ONE_BATCH + 1, t.v['iteration']), t.v['time_delta_iteration'])
            h.myprint('Time per iteration (%d-%d):' % (t.v['iteration'] - settings.ITERATIONS_IN_ONE_BATCH + 1, t.v['iteration']), t.v['time_delta_iteration'] / settings.ITERATIONS_IN_ONE_BATCH)
        else:
            raise Exception("This shouldn't happen.")
        h.myprint('Total time till now:', t.v['time_delta_total'])
        total_time_estimated = t.v['time_delta_total'] / t.v['iteration'] * (settings.ITERATIONS + settings.WARMUP)
        if t.v['iteration'] < (settings.ITERATIONS + settings.WARMUP):
            h.myprint('Estimated total time remaining:', total_time_estimated - t.v['time_delta_total'])
            h.myprint('Estimated total time:', total_time_estimated)










        t.save_history()





    t.previous_save_time = h.now() - 2 * t.s.SAVE_TIME_DELTA_MIN
    t.save_history()





    t.post_sample()

    WORKERS.clear()

    print('Finished')

    return folder_name


def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')
    parser.add_argument("-s", "--settings", default=None)
    parser.add_argument("-p", "--preprocessing", choices=[None, 'normalize_sf', 'autoCorrection'], default=None)
    args = parser.parse_args()



    data_file = args.data_file[0]



    if args.settings is not None:
        settings = h.load_settings(args.settings)
    else:
        from outpyr import settings

    t1 = h.now()
    if os.path.isfile(data_file):
        run(data_file, settings, args.preprocessing)
    else:
        parser.error("The file (%s) you provided does not exist." % data_file)


if __name__ == '__main__':
    main()
