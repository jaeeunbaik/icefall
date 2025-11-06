#!/usr/bin/env python3
"""Test RIR cuts loading"""

from lhotse import CutSet

print("Loading RIR cuts...")
cuts = CutSet.from_file('data/rir/rir_cuts.jsonl.gz')
print(f'Successfully loaded {len(cuts)} RIR cuts')
print(f'Total duration: {cuts.total_duration():.2f} seconds')
print(f'Average duration: {cuts.total_duration()/len(cuts):.3f} seconds')

# Check first cut structure
first_cut = list(cuts)[0]
print(f'\nFirst cut ID: {first_cut.id}')
print(f'First cut duration: {first_cut.duration}')
print(f'Has recording: {first_cut.recording is not None}')

if first_cut.recording:
    print(f'Recording ID: {first_cut.recording.id}')
    print(f'Recording num_samples: {first_cut.recording.num_samples}')
    print(f'Recording duration: {first_cut.recording.duration}')
    print(f'Recording sampling_rate: {first_cut.recording.sampling_rate}')
else:
    print('WARNING: Recording is None!')
    print(f'Cut attributes: {vars(first_cut)}')
