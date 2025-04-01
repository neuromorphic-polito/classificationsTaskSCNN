import sys
sys.path.append('../../')
import os


##############################
# ##### Spike encoding ##### #
##############################
filterbanks = ['butterworth', 'gammatone']
channels = [32, 64]
for filterbank in filterbanks:
    for channel in channels:
        command = f'python -u 1-encodingSpike.py -f={filterbank} -c={channel}'
        print(command)
        os.system(command)


###################################
# ##### Sonogram generation ##### #
###################################
##### Encoding algorithm #####
encodings = ['RATE', 'TBR', 'SF', 'ZCSF', 'MW', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

##### Binning settings #####
configurations = [
    ('butterworth', 32, 31.25),
    ('gammatone', 32, 31.25),

    ('butterworth', 32, 20),  # (32, 70) big bins  (32, 10) small bins
    ('gammatone', 32, 20),

    ('butterworth', 64, 20),
    ('gammatone', 64, 20),

    ('butterworth', 64, 15.625),
    ('gammatone', 64, 15.625),
]

for encoding in encodings:
    for configuration in configurations:
        filterbank, channel, binsWindow = configuration
        command = f'python -u 2-encodingSonogram.py -e={encoding} -f={filterbank} -c={channel} -b={binsWindow}'
        print(command)
        os.system(command)
