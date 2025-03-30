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
quantile = ['0', '25', '50', '75']

##### Run training models #####
for subset in subsets:
    for encoding in encodings:
        for configuration in configurations:
            filterbank, channel, bins = configuration
            for structure in structures:
                for quartile in quantile:
                    command = f'python -u 1-inferenceSCNN-CompleteReducted.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -r={quartile}'
                    print(command)
                    os.system(command)

##### Run training models #####
quantile = ['median', 'upper']
count = 1
for subset in subsets:
    for encoding in encodings:
        for configuration in configurations:
            filterbank, channel, bins = configuration
            for structure in structures:
                for quartile in quantile:
                    command = f'python -u 2-inferenceSCNN-Pruned.py -n={subset} -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile}'
                    print(command)
                    os.system(command)
