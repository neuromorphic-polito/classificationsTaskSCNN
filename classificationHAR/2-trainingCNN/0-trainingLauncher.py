import os


###############################################
# ##### Configuration space definitions ##### #
###############################################
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
quartiles = ['median', 'upper']
trials = 60

##### Run training models #####
count = 1
for subset in subsets:
    for encoding in encodings:
        for configuration in configurations:
            filterbank, channel, bins = configuration
            for structure in structures:
                os.system(f'python 1-trainingCNN-Complete.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -t={trials}')
                for quartile in quartiles:
                    os.system(f'python 2-trainingCNN-Pruned.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile} -t={trials}')
                    if count % 10 == 0:
                        file = open('nohup.out', 'w')
                        file.close()
                    count += 1
