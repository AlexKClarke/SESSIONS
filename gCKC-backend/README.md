gCKC-software backend.

To separate surface EMG, set self.intramuscular = False (this influences the filtering cut-offs). To only high-pass the 
intramuscular EMG, set self.high_pass_intra = True
To use source deflation, set self.source_def = True, otherwise it will use peel off.

The packages needed to run this programme are in requirements.txt
The main function is core.py, the source separation functions are in helpers.py and the data is loaded in readers.py
