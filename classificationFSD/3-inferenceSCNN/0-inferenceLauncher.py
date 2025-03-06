import os

###############################################
# ##### Configuration space definitions ##### #
###############################################
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
quartiles = ['000000', '110000', '111000', '111100']

##### Run training models #####
count = 1
for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, bins = configuration
        for structure in structures:
            for quartile in quartiles:
                os.system(f'python -u 1-inferenceSCNN-CompleteReducted.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -r={quartile}')
                if count % 10 == 0:
                    file = open('nohup.out', 'w')
                    file.close()
                count += 1

##### Run training models #####
quartiles = ['median', 'upper']
count = 1
for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, bins = configuration
        for structure in structures:
            for quartile in quartiles:
                os.system(f'python -u 2-inferenceSCNN-Pruned.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile}')
                if count % 10 == 0:
                    file = open('nohup.out', 'w')
                    file.close()
                count += 1
