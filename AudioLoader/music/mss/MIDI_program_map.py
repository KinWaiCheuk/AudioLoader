import pickle
import csv
from pathlib import Path

path = Path(__file__).parent.joinpath("MIDI_program_map.tsv") 
print(f"{Path(__file__).parent}")
with path.open() as csv_file:

# with open('./End2End/MIDI_program_map.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    idx2class_name = {}
    idx2instrument_name = {}
    idx2instrument_class = {}
    for row in csv_reader:
        idx2class_name[int(row[0])] = row[1]
        idx2instrument_name[int(row[0])] = row[2]
        idx2instrument_class[int(row[0])] = row[3]     
        
path = Path(__file__).parent.joinpath("slakh_instruments.pkl")        
# slakh_instrument = pickle.load(open('./End2End/slakh_instruments.pkl', 'rb'))
slakh_instrument = pickle.load(path.open('rb'))
MIDIProgramName2class_idx = {}
class_idx2MIDIProgramName = {}

W_MIDIClassName2class_idx = {}
W_class_idx2MIDIClass = {}

MIDIClassName2class_idx = {}
class_idx2MIDIClass = {}
for idx,i in enumerate(slakh_instrument):
    MIDIProgramName2class_idx[idx2instrument_name[i]] = idx
    class_idx2MIDIProgramName[idx] = idx2instrument_name[i]
    
    W_MIDIClassName2class_idx[idx2instrument_class[i]] = idx
    W_class_idx2MIDIClass[idx] = idx2instrument_class[i]
    
# # Assigning Empty class    
# MIDIProgramName2class_idx['empty'] = idx+1

# More general definition    
unique_instrument_class = []
for i in idx2instrument_class.items():
    if i[1] in unique_instrument_class:
        continue
    else:
        unique_instrument_class.append(i[1])
    
for idx, class_name in enumerate(unique_instrument_class):
    MIDIClassName2class_idx[class_name] = idx
    class_idx2MIDIClass[idx] = class_name

# assign empty class
MIDIClassName2class_idx['Empty'] = idx+1
class_idx2MIDIClass[idx+1] = 'Empty'
    
MIDI_PROGRAM_NUM = len(MIDIProgramName2class_idx)
MIDI_Class_NUM = len(MIDIClassName2class_idx)
W_MIDI_Class_NUM = len(W_class_idx2MIDIClass)