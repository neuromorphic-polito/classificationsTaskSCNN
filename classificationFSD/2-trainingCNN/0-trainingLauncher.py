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
quartiles = ['median', 'upper']
trials = 30

##### Run training models #####
count = 1
for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, bins = configuration
        for structure in structures:
            os.system(f'python 1-trainingCNN-Complete.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -t={trials}')
            for quartile in quartiles:
                os.system(f'python 2-trainingCNN-Pruned.py -e={encoding} -f={filterbank} -c={channel} -b={bins} -s={structure} -q={quartile} -t={trials}')
                if count % 10 == 0:
                    file = open('nohup.out', 'w')
                    file.close()
                count += 1
