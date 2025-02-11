# calculate LZ complexity of the same sequence used for learning abstraction
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from collections import Counter

def lzcompression(seq):
    def lz_complexity(sequence):
        seql = 0 # length of sequence after compression
        parsed_sequence = []
        n = len(sequence)
        phrases = dict()
        i = 0
        complexity = 0

        while i < n:
            j = i + 1
            while j <= n:
                current_phrase = sequence[i:j]
                if tuple(current_phrase) not in phrases:
                    phrases[tuple(current_phrase)] = 1
                    break
                else:
                    phrases[tuple(current_phrase)] += 1
                j += 1

            parsed_sequence.append(tuple(current_phrase))
            seql = seql + 1  # increment size of the parsed sequence

            i = j
        return phrases, seql, parsed_sequence

    # Flatten the array to 1D
    flattened_array = seq.ravel().astype(int)

    # Convert the 1D array to a string of characters
    array_string = ''.join(map(str, flattened_array))
    # Convert seq as np array into string:
    sequence = array_string
    print('before lz compression ', sequence, ' \n sequence length', len(array_string))
    phrases, seql, parsed_sequence = lz_complexity(sequence)
    print('after lz compression ', parsed_sequence)
    print(phrases)
    print('total sequence length after parsing ', sum([len(item) for item in parsed_sequence]))
    print('length of parsed sequence ', len(parsed_sequence))
    count_freq = Counter(parsed_sequence) # how often each word is being parsed
    freq = np.array(list(count_freq.values()))
    complexity = 0
    for k in count_freq:
        count_freq[k] = count_freq[k]/freq.sum()
    ps = freq / freq.sum()
    storage = -np.sum(np.log2(ps)) # storage cost of all chunks
    for k in parsed_sequence:
        complexity = complexity - np.log2(count_freq[k])

    print(f"Sequence Complexity by LZ: {complexity} bits")
    return complexity, seql, storage




def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def plot_learning_comparison(datann, sz = 10, savename = 'modelcomparisonall.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    datahcm = np.load('./data/hcm'+ ' sz = ' + str(sz) + '.npy')
    datahvm = np.load('./data/hvm'+ ' sz = ' + str(sz) + '.npy')

    # both are three dimensional arrays

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']

    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(datahcm[0,:, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        hcm_mean = np.mean(datahcm[:, :, i + 1], axis=0)
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis=0)
        nn_mean = np.mean(datann[:, :, 0], axis=0)

        ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha=0.3)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha=0.3)
        if i == 3:
            ax.plot(x,nn_mean, label='LZ', color='gray', linewidth=4, alpha=0.3)
            for j in range(0, datann.shape[0]):
                ax.plot(x, datann[j, :, 0], color='gray', linewidth=1, alpha=0.3)

        for j in range(0, datahcm.shape[0]):
            ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha=0.3)
            ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha=0.3)

        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig(savename)

    return


def plot_alphabet_increase_progression(savename = './data/alphabet_increase_progression.png'):
    hcm = []
    hvm = []
    lz = []
    nn = []

    hcm_se = []
    hvm_se = []
    lz_se = []
    nn_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    for sz in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm' + ' sz = ' + str(sz) + '.npy')
        datahvm = np.load('./data/hvm' + ' sz = ' + str(sz) + '.npy')
        datann = np.load('./data/nn' + ' sz = ' + str(sz) + '.npy')
        datalz = np.load('./data/lz' + ' sz = ' + str(sz) + '.npy')

        hcm_mean_complexity = np.mean(datahcm[:, -1, 4], axis=0)# average over different runs
        hvm_mean_complexity = np.mean(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_mean_complexity = np.mean(datalz[:, -1, 0], axis=0)# average over different runs
        nn_mean_complexity = np.mean(datann[:, -1, 0], axis=0)# average over different runs

        # calculate standard error
        hcm_se_complexity = stats.sem(datahcm[:, -1, 4], axis=0)
        hvm_se_complexity = stats.sem(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_se_complexity = stats.sem(datalz[:, -1, 0], axis=0)
        nn_se_complexity = stats.sem(datann[:, -1, 0], axis=0)# average over different runs

        hcm.append(hcm_mean_complexity)
        hvm.append(hvm_mean_complexity)
        lz.append(lz_mean_complexity)
        nn.append(nn_mean_complexity)

        hcm_se.append(hcm_se_complexity)
        hvm_se.append(hvm_se_complexity)
        lz_se.append(lz_se_complexity)
        nn_se.append(nn_se_complexity)

    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")

    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, yerr=lz_se, label='LZ', color='gray', linewidth=3, fmt = '-o')
    plt.errorbar(x, nn, yerr=nn_se, label='NN', color='green', linewidth=3, fmt = '-o')

    plt.xlabel('Alphabet size')
    plt.ylabel('Sequence Complexity (bits)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return




def plot_depth_increase_progression(savename = './data/depth_increase_progression.png'):
    hcm = []
    hvm = []
    lz = []
    nn = []

    hcm_se = []
    hvm_se = []
    lz_se = []
    nn_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    for sz in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(sz) + '.npy')
        datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(sz) + '.npy')
        datann = np.load('./data/nn_fixed_support_set' + ' d = ' + str(sz) + '.npy')
        datalz = np.load('./data/lz_fixed_support_set' + ' d = ' + str(sz) + '.npy')

        hcm_mean_complexity = np.mean(datahcm[:, -1, 4], axis=0)# average over different runs
        hvm_mean_complexity = np.mean(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_mean_complexity = np.mean(datalz[:, -1, 0], axis=0)# average over different runs
        nn_mean_complexity = np.mean(datann[:, -1, 0], axis=0)# average over different runs

        # calculate standard error
        hcm_se_complexity = stats.sem(datahcm[:, -1, 4], axis=0)
        hvm_se_complexity = stats.sem(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_se_complexity = stats.sem(datalz[:, -1, 0], axis=0)
        nn_se_complexity = stats.sem(datann[:, -1, 0], axis=0)# average over different runs

        hcm.append(hcm_mean_complexity)
        hvm.append(hvm_mean_complexity)
        lz.append(lz_mean_complexity)
        nn.append(nn_mean_complexity)

        hcm_se.append(hcm_se_complexity)
        hvm_se.append(hvm_se_complexity)
        lz_se.append(lz_se_complexity)
        nn_se.append(nn_se_complexity)

    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")

    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, yerr=lz_se, label='LZ', color='gray', linewidth=3, fmt = '-o')
    plt.errorbar(x, nn, yerr=nn_se, label='NN', color='green', linewidth=3, fmt = '-o')

    plt.xlabel('Depth')
    plt.ylabel('Sequence Complexity (bits)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return






def plot_explanatory_volume_per_storage(savename = './data/explanatory_volume_per_storage.png'):
    hcm = []
    hvm = []
    lz = []
    gt = []

    x = [1, 10, 20, 30, 40, 50, 60]
    sz = 10

    for d in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datalzstorage = np.load('./data/lz_storage' + ' d = ' + str(d) + '.npy')
        GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(sz) + '.npy')
        datalz_encoding_bits = np.load('./data/lz_encoding_bits' + ' d = ' + str(d) + '.npy')

        hcm_storage = np.mean(datahcm[:, -1, 8], axis=0)# average over different runs
        hvm_storage = np.mean(datahvm[:, -1, 8], axis=0)# average over different runs
        lz_storage = np.mean(datalzstorage[:, -1, 0], axis=0)
        gt_storage = GT[0,8]

        hcm_mean_explanatory_volume = np.mean(datahcm[:, -1, 3], axis=0)# average over different runs
        hvm_mean_explanatory_volume = np.mean(datahvm[:, -1, 3], axis=0)# average over different runs
        lz_mean_explanatory_volume = 1000/np.mean(datalz_encoding_bits[:, -1, 0], axis=0)/12# average over different runs
        gt_explanatory_volume = GT[0, 3] # the number of symbols to encode sequences in ground truth

        hcm.append(hcm_mean_explanatory_volume/hcm_storage)
        hvm.append(hvm_mean_explanatory_volume/hvm_storage)
        lz.append(lz_mean_explanatory_volume/lz_storage)
        gt.append(gt_explanatory_volume/gt_storage)


    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")

    plt.errorbar(x, hcm, label='HCM', color='orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, label='HVM', color='blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, label='LZ', color='gray', linewidth=3, fmt = '-o')
    plt.errorbar(x, gt, label='GT', color='black', linewidth=3, fmt = '-o')

    plt.xlabel('Depth')
    plt.ylabel('Explanatory Volume Per Storage Unit (l/bit)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return



def plot_lz_comparison_complexity(savename = './data/depth_increase_complexity.png'):
    hcm = []
    hvm = []
    lz = []
    gt = []

    hcm_se = []
    hvm_se = []
    lz_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    bit_per_symbol = 12
    sz = 10

    for d in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datalz_encoding_bits = np.load('./data/lz_encoding_bits' + ' d = ' + str(d) + '.npy')
        datalz_complexity = np.load('./data/lz_complexity' + ' d = ' + str(d) + '.npy')
        data_gt_encoding_bits = np.load('./data/sequence_original_encoding_bits' + ' sz = ' + str(d) + '.npy')
        GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(sz) + '.npy')
        data_gt_complexity = [GT[0, 4]]

        hcm_mean_complexity = np.mean(datahcm[:, -1, 4], axis=0)# average over different runs
        hvm_mean_complexity = np.mean(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_mean_complexity = np.mean(datalz_complexity[:, -1, 0], axis=0)# average over different runs
        gt_mean_complexity = data_gt_complexity[0]

        # calculate standard error
        hcm_se_complexity = stats.sem(datahcm[:, -1, 4], axis=0)
        hvm_se_complexity = stats.sem(datahvm[:, -1, 4], axis=0)# average over different runs
        lz_se_complexity = stats.sem(datalz_complexity[:, -1, 0], axis=0)

        hcm.append(hcm_mean_complexity)
        hvm.append(hvm_mean_complexity)
        lz.append(lz_mean_complexity)
        gt.append(gt_mean_complexity)

        hcm_se.append(hcm_se_complexity)
        hvm_se.append(hvm_se_complexity)
        lz_se.append(lz_se_complexity)

    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")
    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, yerr=lz_se, label='LZ', color='gray', linewidth=3, fmt = '-o')
    plt.errorbar(x, gt, label='GT', color='black', linewidth=3, fmt = '-o')

    plt.xlabel('Depth')
    plt.ylabel('Sequence Complexity (bits)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return



def plot_lz_comparison_seql(savename = './data/depth_increase_seql.png'):
    hcm = []
    hvm = []
    lz = []
    gt = []
    raw = []


    hcm_se = []
    hvm_se = []
    lz_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    bit_per_symbol = 12


    for d in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datalz_encoding_bits = np.load('./data/lz_encoding_bits' + ' d = ' + str(d) + '.npy')
        data_raw_encoding_bits = np.load('./data/sequence_original_encoding_bits' + ' sz = ' + str(d) + '.npy') # sorry I forget to change the alphabet size to d
        GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(10) + '.npy')
        data_gt_encoding_bits = 1000/GT[0, 3]*bit_per_symbol # the number of symbols to encode sequences in ground truth

        hcm_mean_encoding_bits = 1000/np.mean(datahcm[:, -1, 3], axis=0)*bit_per_symbol# average over different runs
        hvm_mean_encoding_bits = 1000/np.mean(datahvm[:, -1, 3], axis=0)*bit_per_symbol# average over different runs
        lz_mean_encoding_bits = np.mean(datalz_encoding_bits[:, -1, 0], axis=0)# average over different runs
        raw_mean_encoding_bits = np.mean(data_raw_encoding_bits[:, -1, 0], axis=0)#

        # calculate standard error
        hcm_se_encoding_bits = stats.sem(1000/datahcm[:, -1, 3]*bit_per_symbol, axis=0)
        hvm_se_encoding_bits = stats.sem(1000/datahvm[:, -1, 3]*bit_per_symbol, axis=0)# average over different runs
        lz_se_encoding_bits = stats.sem(datalz_encoding_bits[:, -1, 0], axis=0)

        hcm.append(hcm_mean_encoding_bits)
        hvm.append(hvm_mean_encoding_bits)
        lz.append(lz_mean_encoding_bits)
        raw.append(raw_mean_encoding_bits)
        gt.append(data_gt_encoding_bits)

        hcm_se.append(0)#hcm_se_encoding_bits)
        hvm_se.append(0)#hvm_se_encoding_bits)
        lz_se.append(0)#lz_se_encoding_bits)

    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")
    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, yerr=lz_se, label='LZ', color='gray', linewidth=3, fmt = '-o')
    plt.errorbar(x, raw, label='Uncompressed', color='green', linewidth=3, fmt = '-o')
    plt.errorbar(x, gt, label='Generative Model', color='black', linewidth=3, fmt = '-o')

    plt.xlabel('d')
    plt.ylabel('Encoding Length (bits)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return




def plot_lz_comparison_seql_alphabet_increase(savename = './data/alphabet_increase_seql.png'):
    hcm = []
    hvm = []
    lz = []

    hcm_se = []
    hvm_se = []
    lz_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    bit_per_symbol = 12

    raw_seql = 1000
    for sz in [1, 10, 20, 30, 40, 50, 60]:
        datahcm = np.load('./data/hcm' + ' sz = ' + str(sz) + '.npy')
        datahvm = np.load('./data/hvm' + ' sz = ' + str(sz) + '.npy')
        datalz_encoding_bits = np.load('./data/lz_seql' + ' sz = ' + str(sz) + '.npy')

        hcml = raw_seql/np.mean(datahcm[:, -1, 3], axis=0)
        hvml = raw_seql/np.mean(datahvm[:, -1, 3], axis=0)
        hcm_mean_encoding_bits = raw_seql/np.mean(datahcm[:, -1, 3], axis=0)*bit_per_symbol# average over different runs
        hvm_mean_encoding_bits = raw_seql/np.mean(datahvm[:, -1, 3], axis=0)*bit_per_symbol# average over different runs
        lz_mean_encoding_bits = np.mean(datalz_encoding_bits[:, -1, 0], axis=0)# average over different runs

        # calculate standard error
        hcm_se_encoding_bits = stats.sem(raw_seql/datahcm[:, -1, 3]*bit_per_symbol, axis=0)
        hvm_se_encoding_bits = stats.sem(raw_seql/datahvm[:, -1, 3]*bit_per_symbol, axis=0)# average over different runs
        lz_se_encoding_bits = stats.sem(datalz_encoding_bits[:, -1, 0], axis=0)

        hcm.append(hcm_mean_encoding_bits)
        hvm.append(hvm_mean_encoding_bits)
        lz.append(lz_mean_encoding_bits)

        hcm_se.append(hcm_se_encoding_bits)
        hvm_se.append(hvm_se_encoding_bits)
        lz_se.append(lz_se_encoding_bits)

    # Setting a style
    sns.set(style="white")

    # Better color palette
    colors = sns.color_palette("pastel")
    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')
    plt.errorbar(x, lz, yerr=lz_se, label='LZ', color='gray', linewidth=3, fmt = '-o')

    plt.xlabel('Alphabet size')
    plt.ylabel('Encoding Length (bits)')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return




def plot_depth_hvm_hcm_comparison(savename = './data/depth_increase_comparison_hvm_hcm.png'):
    hcm = []
    hvm = []

    hcm_se = []
    hvm_se = []

    x = [1, 10, 20, 30, 40, 50, 60]

    for d in [1, 10, 20, 30, 40, 50, 60]:# increasing depth
        datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
        datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')

        # count the number of chunks
        hcm_mean_nc = np.mean(datahcm[:, -1, 6], axis=0)# average over different runs
        hvm_mean_nc = np.mean(datahvm[:, -1, 6], axis=0)# average over different runs

        # calculate standard error
        hcm_se_nc = stats.sem(datahcm[:, -1, 6], axis=0)
        hvm_se_nc = stats.sem(datahvm[:, -1, 6], axis=0)# average over different runs

        hcm.append(hcm_mean_nc)
        hvm.append(hvm_mean_nc)

        hcm_se.append(hcm_se_nc)
        hvm_se.append(hvm_se_nc)

    # Setting a style
    sns.set(style="whitegrid")

    plt.errorbar(x, hcm, yerr=hcm_se, label='HCM', color= 'orange', linewidth=3, fmt = '-o')
    plt.errorbar(x, hvm, yerr=hvm_se, label='HVM', color= 'blue', linewidth=3, fmt = '-o')

    plt.xlabel('Depth')
    plt.ylabel('N Chunks')
    # Show the figure
    plt.legend(loc='best')
    # save the figure
    plt.savefig(savename)
    return






def plot_key_model_comparison(d = None, savename = 'barmodelcomparison.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    # both are three dimensional arrays

    titles = ['parsing length', 'explanatory volume', 'sequence complexity','representation entropy']
    # 1, 3, 4, 5
    units = ['n chunk', 'l', 'bits', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(1, 4, figsize=(10, 6))

    datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datann = np.load('./data/nn_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datalz = np.load('./data/lz_fixed_support_set' + ' d = ' + str(d) + '.npy')

    categories = ['HCM', 'HVM', 'Generative Model']

    gt = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(5) + '.npy')

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))  # 1 row, 4 columns

    plot_idx = 0
    for i in [1, 3, 4, 5]:
        hcm_mean = np.mean(datahcm[:, -1, i], axis=0) # at the end of learning
        hvm_mean = np.mean(datahvm[:, -1, i], axis=0)
        gt_mean = gt[0, i]
        values = [hcm_mean, hvm_mean, gt_mean]

        axs[plot_idx].bar(categories, values, color = ['orange', 'blue', 'green'])
        axs[plot_idx].set_title(titles[plot_idx])
        axs[plot_idx].set_ylabel(units[plot_idx])

        plot_idx = plot_idx + 1


    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig(savename)

    return




def eval_lz_encoding_bits(savename = './data/eval_lz_encoding_bit.png'):
    bit_per_symbol = 12
    size_increment = [1, 10, 20, 30, 40, 50, 60]
    for sz in size_increment:
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 1000
        n_measure = 1 # just measure the sequence complexity
        n_iter = int(len(fullseq)/slice_sz)
        datalz_encoding_bits = np.empty((n_iter, 10, n_measure)) # 10: number of iterations (epoch)
        sequence_original_encoding_bits = np.empty((n_iter, 10, n_measure))
        datalz_complexity = np.empty((n_iter, 10, n_measure)) # 10: number of iterations (epoch)
        datalz_storage = np.empty((n_iter, 10, n_measure))

        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            # lz compression complexity (about constant)
            complexity, seql, storage = lzcompression(seq)
            print('seql after compression ', seql)
            datalz_encoding_bits[i, :, :] = np.array(seql*bit_per_symbol)
            datalz_complexity[i, :, :] = np.array(complexity)
            datalz_storage[i,:,:] = np.array(storage)
            sequence_original_encoding_bits[i, :, :] = np.array(len(seq)*bit_per_symbol)

            i = i + 1
        np.save('./data/lz_encoding_bits' + ' d = ' + str(sz) + '.npy', datalz_encoding_bits)
        np.save('./data/lz_complexity' + ' d = ' + str(sz) + '.npy', datalz_complexity)
        np.save('./data/lz_storage' + ' d = ' + str(sz) + '.npy', datalz_storage)
        np.save('./data/sequence_original_encoding_bits' + ' sz = ' + str(sz) + '.npy', sequence_original_encoding_bits)

        #plot_learning_comparison(datalz, sz=sz, savename='./data/lz_encoding_bits' + ' sz = ' + str(sz) + '.png')
    return





def eval_lz_encoding_bits_longer_sequences(seql=2000):
    bit_per_symbol = 12
    sequence_length = seql
    size_increment = [30]
    for sz in size_increment:
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'

        #openpath = './generative_sequences_different_parameters/random_abstract_sequence_a='+ str(10) + '_d='+ str(sz) + '_p_variable=' + str(0.5) +'_seql=' + str(sequence_length) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = sequence_length
        n_measure = 1 # just measure the sequence complexity
        n_iter = int(len(fullseq)/slice_sz)
        datalz_encoding_bits = np.empty((n_iter, 10, n_measure)) # 10: number of iterations (epoch)
        sequence_original_encoding_bits = np.empty((n_iter, 10, n_measure))
        datalz_complexity = np.empty((n_iter, 10, n_measure)) # 10: number of iterations (epoch)
        datalz_storage = np.empty((n_iter, 10, n_measure))

        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            # lz compression complexity (about constant)
            complexity, seql, storage = lzcompression(seq)
            print('seql after compression ', seql)
            datalz_encoding_bits[i, :, :] = np.array(seql*bit_per_symbol)
            datalz_complexity[i, :, :] = np.array(complexity)
            datalz_storage[i,:,:] = np.array(storage)
            sequence_original_encoding_bits[i, :, :] = np.array(len(seq)*bit_per_symbol)

            i = i + 1
        np.save('./data/lz_encoding_bits_d='+ sz +'_seql=' + str(sequence_length) + '.npy', datalz_encoding_bits)
        np.save('./data/lz_complexity_d='+ sz +'_seql=' + str(sequence_length) + '.npy', datalz_complexity)
        np.save('./data/lz_storage_d='+ sz +'_seql=' + str(sequence_length) + '.npy', datalz_storage)
        np.save('./data/sequence_original_encoding_bits_d='+sz +'_seql=' + str(sequence_length) + '.npy', sequence_original_encoding_bits)

    return

def eval_lz_size_increment():
    size_increment = [5, 10, 15, 20, 25, 30, 35]
    for sz in size_increment:
        openpath = './generative_sequences/random_abstract_sequence' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 1000
        n_measure = 1 # just measure the sequence complexity
        n_iter = int(len(fullseq)/slice_sz)
        bitpersymbol = 12
        datalzcomplexity = np.empty((n_iter, 14, n_measure))
        datalzseqencodingbits = np.empty((n_iter, 14, n_measure))
        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            # lz compression complexity (about constant)
            #complexity, seql = np.array(lzcompression(seq))
            print(len(seq))
            complexity, seql = lzcompression(seq)
            datalzcomplexity[i, :, :] = np.array(complexity)
            datalzseqencodingbits[i,:,:] = np.array(seql)*bitpersymbol
            i = i + 1
        np.save('./data/lz_complexity' + ' sz = ' + str(sz) + '.npy', datalzcomplexity)
        np.save('./data/lz_seql' + ' sz = ' + str(sz) + '.npy', datalzseqencodingbits)

        #plot_learning_comparison(datalz, sz=sz, savename='./data/lz' + ' sz = ' + str(sz) + '.png')
    return


def eval_lz_depth_increment(slice_sz = 1000):
    depth_increment = [1, 10, 20, 30, 40, 50, 60]
    for sz in depth_increment:
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'
        # generative model with increasing depth
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        n_measure = 1 # just measure the sequence complexity
        n_iter = int(len(fullseq)/slice_sz)
        datalz = np.empty((n_iter, 14, n_measure))
        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            # lz compression complexity (about constant)
            datalz[i, :, :] = np.array(lzcompression(seq))
            i = i + 1
        np.save('./data/lz_fixed_support_set' + ' d = ' + str(sz) + '.npy', datalz)
    return




# def plot_model_learning_progression(savename = 'modelcomparison.png'):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # both are three dimensional arrays
#
#     titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
#               'representation entropy', 'n chunks', 'n variables', 'storage cost']
#
#     units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
#     # Create a figure and subplots with 2 rows and 3 columns
#     fig, axs = plt.subplots(2, 4, figsize=(10, 6))
#     x = np.cumsum(datahcm[0,:, 0])
#
#     for i, ax in enumerate(axs.flat):
#         if i >= 8:
#             break
#         hcm_mean = np.mean(datahcm[:, :, i + 1], axis = 0)
#         hvm_mean = np.mean(datahvm[:, :, i + 1], axis = 0)
#         ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha = 0.3)
#         ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha = 0.3)
#         for j in range(0, datahcm.shape[0]):
#             ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha = 0.3)
#             ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha = 0.3)
#
#         ax.set_title(titles[i])
#         ax.set_ylabel(units[i])
#         ax.set_xlabel('Sequence Length')
#     # Adjust spacing between subplots
#     fig.tight_layout()
#     # Show the figure
#     plt.legend()
#     plt.show()
#     # save the figure
#     fig.savefig(savename)
#
#     return

def efficiency_curve_lzw(d = 20, makeplots = True):
    # take a sequence
    overhead_char = 256 # to cover all 256 characters log_2(256) is the number of bits
    openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(d) + '.npy'
    with open(openpath, 'rb') as f:
        seq = np.load(f)

    def lz_complexity(sequence):
        n_entries_dict = [] # number of entries in the dictionary (excluding overhead)
        parsed_seql_record = [] # progress in encoding the length of the sequence

        seql = 0 # length of sequence after compression
        parsed_sequence = []
        n = len(sequence)
        phrases = dict()
        i = 0
        while i < n:
            j = i + 1
            while j <= n:
                current_phrase = sequence[i:j]
                if tuple(current_phrase) not in phrases:
                    phrases[tuple(current_phrase)] = 1
                    break
                else:
                    phrases[tuple(current_phrase)] += 1
                j += 1

            parsed_sequence.append(tuple(current_phrase))
            seql = seql + 1  # increment size of the parsed sequence
            n_entries_dict.append(seql)
            parsed_seql_record.append(i)

            i = j
        return n_entries_dict, parsed_seql_record

    # Flatten the array to 1D
    flattened_array = seq.ravel().astype(int)

    # Convert the 1D array to a string of characters
    array_string = ''.join(map(str, flattened_array))
    n_entries_dict, parsed_seql_record = lz_complexity(array_string)
    n_entries_dict = np.array(n_entries_dict) + overhead_char
    parsed_seql_record = np.array(parsed_seql_record)
    lzcodingefficiency = n_entries_dict/parsed_seql_record
    if makeplots:
        plt.plot(parsed_seql_record, lzcodingefficiency, '-', color ='#36454F')
        plt.ylabel('N dictionary entries per sequence unit')
        plt.xlabel('sequence length')
        plt.title('Coding Efficiency')


    ###############
    datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(10) + '.npy')
    x = np.cumsum(datahcm[0,:, 0])

    hcm_mean_n_chunk = np.mean(datahcm[:, :, 6], axis=0)  # at the end of training
    gt_n_chunk = GT[0, 6]  # the number of symbols to encode sequences in ground truth
    hvm_mean_n_chunk = np.mean(datahvm[:, :, 6], axis=0)   # average over different runs
    hcm_mean_coding_efficiency = hcm_mean_n_chunk/x
    hvm_mean_coding_efficiency = hvm_mean_n_chunk/x
    gt_coding_efficiency = gt_n_chunk/x
    if makeplots:
        plt.plot(x, hcm_mean_n_chunk/x, '-', color = '#CC5500')
        plt.plot(x, hvm_mean_n_chunk/x, '-', color = 'royalblue')

        plt.yscale('log')
        plt.plot(x, gt_n_chunk/x, '-', color = 'forestgreen')

        plt.legend(['LZ78','HCM','HVM','GT'])

    return parsing_seql_record, lzcodingefficiency, hvm_mean_coding_efficiency, hcm_mean_coding_efficiency, gt_coding_efficiency

def trade_off_interplay(d = 30):
    x = []
    y = []
    z = []
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    seql = 1000
    representation_complexity = np.mean(datahvm[:, :, 2], axis=0)  # parsed sequence length averaged over different runs
    sequence_length = seql / np.mean(datahvm[:, :, 3], axis=0)  # parsed sequence length averaged over different runs
    entropy = np.mean(datahvm[:, :, 5], axis=0)  # parsed sequence length averaged over different runs
    storage = np.mean(datahvm[:, :, 8], axis=0)  # parsed sequence length averaged over different runs
    # sequence_complexity = seql / np.mean(datahvm[:, :, 4],axis=0)  # parsed sequence length averaged over different runs

    # Sample data
    x = x + list(representation_complexity)
    y = y + list(sequence_length)
    z = z + list(entropy)  # Additional dimension for coloring

    # Create the plot
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Blues')  # Choosing the colormap

    # Create a color reference to the third dimension
    normalize = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    colors = cmap(normalize(z))

    # Plotting the line with color changes
    for i in range(len(x) - 1):
        print(x[i:i+1], y[i:i+1], colors[i])
        ax.plot(x[i:i+1], y[i:i+1], 'o', color=colors[i], markeredgecolor = 'black')

    # Adding a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Entropy')
    ax.set_xlabel('Representation Complexity')
    ax.set_ylabel('Sequence Length')

    # Show the plot
    plt.show()

    return


def calculate_sem(data):
    """
    Calculate the standard error of the mean (SEM) for a given array of data.

    Parameters:
        data (numpy.ndarray): A numpy array containing the data points.

    Returns:
        float: The standard error of the mean of the data.
    """
    # Calculate the standard deviation of the data
    std_dev = np.std(data, ddof=1)  # ddof=1 provides an unbiased estimator by using N-1 in the denominator

    # Calculate the number of observations in the data
    n = len(data)

    # Calculate SEM
    sem = std_dev / np.sqrt(n)

    return sem

def sequence_complexity_comparison():
    # bar plot comparison of sequence complexity comparison between different encoding algorithms
    d = 30
    datahcm = np.load('./data/hcm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datahvm = np.load('./data/hvm_fixed_support_set' + ' d = ' + str(d) + '.npy')
    datalz_encoding_bits = np.load('./data/lz_encoding_bits' + ' d = ' + str(d) + '.npy')
    data_raw_encoding_bits = np.load('./data/sequence_original_encoding_bits' + ' sz = ' + str(
        d) + '.npy')  # sorry I forget to change the alphabet size to d
    GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(10) + '.npy')
    seql = 1000
    bit_per_symbol = 12

    hcm_mean_seq_l = seql / np.mean(datahcm[:, -1, 3], axis=0)  # at the end of training
    hvm_mean_seq_l = seql / np.mean(datahvm[:, -1, 3], axis=0)   # average over different runs
    lz_mean_seq_l = np.mean(datalz_encoding_bits[:, -1, 0], axis=0)/bit_per_symbol  # average over different runs
    gt_seq_l = seql / GT[0, 3]  # the number of symbols to encode sequences in ground truth
    raw_mean_seq_l = np.mean(data_raw_encoding_bits[:, -1, 0], axis=0)/bit_per_symbol#
    sem_seq_l = [calculate_sem(seql/datahcm[:, -1, 3]), calculate_sem(seql/datahvm[:, -1, 3]),
                      calculate_sem(datalz_encoding_bits[:, -1, 0]/bit_per_symbol), 0]

    datalz_complexity = np.load('./data/lz_complexity' + ' d = ' + str(d) + '.npy')
    GT = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = 10.npy')
    data_gt_complexity = [GT[0, 4]]

    hcm_mean_complexity = np.mean(datahcm[:, -1, 4], axis=0)  # average over different runs
    hvm_mean_complexity = np.mean(datahvm[:, -1, 4], axis=0)  # average over different runs
    lz_mean_complexity = np.mean(datalz_complexity[:, -1, 0], axis=0)  # average over different runs
    gt_mean_complexity = data_gt_complexity[0]
    sem_complexity = [calculate_sem(datahcm[:, -1, 4]), calculate_sem(datahvm[:, -1, 4]), calculate_sem(datalz_complexity[:, -1, 0]), 0]

    data_gt_entropy = [GT[0, 5]]
    hcm_mean_entropy = np.mean(datahcm[:, -1, 5], axis=0)
    hvm_mean_entropy = np.mean(datahvm[:, -1, 5], axis=0)
    gt_mean_entropy = data_gt_entropy[0]
    sem_seq_entropy = [calculate_sem(datahcm[:, -1, 5]), calculate_sem(datahvm[:, -1, 5]), 0]

    plt.rcParams['font.size'] = 18
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 3 rows, 1 column
    # Adjust subplot parameters
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase wspace for more horizontal space
    models = ['Without \n Abstraction', 'With \n Abstraction']
    parsing_length = [np.mean(datahvm[:, -1, 6], axis=0), np.mean(datahvm[:, -1, 1], axis=0)]
    sem_parsing_length = [calculate_sem(datahvm[:, -1, 6]), calculate_sem(datahvm[:, -1, 1])]
    axes[0].bar(models, parsing_length, color=['skyblue','royalblue'], edgecolor = 'black', yerr = sem_parsing_length)
    axes[0].set_ylabel('Parsing Search Steps')  # Bigger font size for y-axis label
    axes[0].set_yscale('log')

    # Sequence Length
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_length = [hcm_mean_seq_l, hvm_mean_seq_l, lz_mean_seq_l, gt_seq_l]
    axes[1].bar(models, seq_length, color=['#CC5500', 'royalblue', '#36454F','forestgreen'], edgecolor = 'black', yerr = sem_seq_l)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Parsing Length |W|')
    plt.yscale('linear')

    # Sequence Complexity
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_complexity = [hcm_mean_complexity, hvm_mean_complexity, lz_mean_complexity, gt_mean_complexity]
    axes[2].bar(models, seq_complexity, color=['#CC5500', 'royalblue', '#36454F', 'forestgreen'], edgecolor = 'black', yerr = sem_complexity)
    axes[2].set_xlabel('Models')
    axes[2].set_ylabel('Sequence Likelihood -logP(S)')
    plt.yscale('linear')

    # Coding efficiency
    models = ['HCM', 'HVM', 'LZ78', 'GT']
    seq_complexity = [hcm_mean_complexity, hvm_mean_complexity, lz_mean_complexity, gt_mean_complexity]
    axes[3].bar(models, seq_complexity, color=['#CC5500', 'royalblue', '#36454F', 'forestgreen'], edgecolor = 'black', yerr = sem_complexity)
    axes[3].set_xlabel('Models')
    axes[3].set_ylabel('')
    plt.yscale('linear')

    # Sequence Complexity
    models = ['HCM', 'HVM', 'GT']
    seq_complexity = [hcm_mean_entropy, hvm_mean_entropy, gt_mean_entropy]
    plt.figure()
    plt.bar(models, seq_complexity, color='royalblue', edgecolor='black', yerr= sem_seq_entropy)
    plt.xlabel('Models')
    plt.ylabel('Parsing Entropy')
    plt.show()
    return



eval_lz_encoding_bits_longer_sequences(seql=4000)

trade_off_interplay(d = 20)

sequence_complexity_comparison()

efficiency_curve_lzw()

######## Explanatory Volume Per Storage

eval_lz_encoding_bits() # this evaluate both the complexity and the depth

plot_explanatory_volume_per_storage()

######## Evaluate Depth Increase

plot_lz_comparison_complexity()

plot_lz_comparison_seql()
plot_key_model_comparison(d = 15) # model comparison of the most typical trend



#plot_depth_hvm_hcm_comparison() # okay this plot is completely weird
#plot_depth_increase_progression()




######## Evaluate Alphabet Increase

eval_lz_size_increment()
plot_lz_comparison_seql_alphabet_increase()

plot_alphabet_increase_progression()








