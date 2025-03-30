import os


###################################
# ##### Configuration space ##### #
###################################
subsets = ['subset1', 'subset2']
##### Encoding algorithm selected #####
encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

##### Binning settings #####
configurations = [
    ('butterworth', 4, 24),
    ('gammatone', 4, 24),

    ('butterworth', 8, 18),
    ('gammatone', 8, 18),

    ('butterworth', 16, 18),
    ('gammatone', 16, 18),
]

structures = ['c06c12f2', 'c12c24f2']
quantile = ['median', 'upper']
trials = 60

##### Training models #####
for subset in subsets:
    for encoding in encodings:
        for configuration in configurations:
            filterbank, channel, bins = configuration
            for structure in structures:
                command = f'python 1-trainingCNN-Complete.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -t={trials}'
                print(command)
                os.system(command)
                for quartile in quantile:
                    command = f'python 2-trainingCNN-Pruned.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile}'
                    print(command)
                    os.system(command)
