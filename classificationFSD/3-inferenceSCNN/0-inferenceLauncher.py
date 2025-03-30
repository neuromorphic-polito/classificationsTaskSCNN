import os

###################################
# ##### Configuration space ##### #
###################################
##### Encoding algorithm selected #####
encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

##### Binning settings #####
configurations = [
    ('butterworth', 32, 50),  # (32, 70)big bins  (32, 10)small bins
    ('gammatone', 32, 50),

    ('butterworth', 32, 32),
    ('gammatone', 32, 32),

    ('butterworth', 64, 50),
    ('gammatone', 64, 50),

    ('butterworth', 64, 64),
    ('gammatone', 64, 64),
]

structures = ['c06c12f2', 'c12c24f2']
quantile = ['0', '25', '50', '75']

##### Run training models #####
for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, bins = configuration
        for structure in structures:
            for quartile in quantile:
                command = f'python -u 1-inferenceSCNN-CompleteReducted.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -r={quartile}'
                print(command)
                os.system(command)

##### Run training models #####
quantile = ['median', 'upper']
for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, bins = configuration
        for structure in structures:
            for quartile in quantile:
                command = f'python -u 2-inferenceSCNN-Pruned.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile}'
                print(command)
                os.system(command)
