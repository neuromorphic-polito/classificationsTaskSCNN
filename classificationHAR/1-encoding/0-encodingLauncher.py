import sys
sys.path.append('../../')
import os


##############################
# ##### Spike encoding ##### #
##############################
subsets = [('subset1', 'A,B,G,H,P,R'), ('subset2', 'G,H,I,J,K,L,M')]
filterbanks = ['butterworth', 'gammatone']
channels = [4, 8, 16]
for subsetName, subsetLabels in subsets:
    for filterbank in filterbanks:
        for channel in channels:
            os.system(f'python -u 1-encodingSpike.py -d=watch -n={subsetName} -s={subsetLabels} -f={filterbank} -c={channel}')
            print(f'python -u 1-encodingSpike.py -d=watch -n={subsetName} -s={subsetLabels} -f={filterbank} -c={channel}')


###################################
# ##### Sonogram generation ##### #
###################################
subsets = [('subset1', 'A,B,G,H,P,R'), ('subset2', 'G,H,I,J,K,L,M')]

##### Encoding algorithm selected #####
encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

##### Binning settings #####
configurations = [
    ('butterworth', 4, 1000/24),  # (32, 70) big bins  (32, 10) small bins
    ('gammatone', 4, 1000/24),

    ('butterworth', 8, 1000/18),
    ('gammatone', 8, 1000/18),

    ('butterworth', 16, 1000/18),
    ('gammatone', 16, 1000/18),
]

for subsetName, _ in subsets:
    for encoding in encodings:
        for configuration in configurations:
            filterbank, channel, binsWindow = configuration
            os.system(f'python -u 2-encodingSonogram.py -n={subsetName} -e={encoding} -f={filterbank} -c={channel} -b={binsWindow}')
            print(f'python -u 2-encodingSonogram.py -n={subsetName} -e={encoding} -f={filterbank} -c={channel} -b={binsWindow}')
