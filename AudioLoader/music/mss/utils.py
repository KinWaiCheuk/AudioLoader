import numpy as np
import collections

class TargetProcessor:
    def __init__(
        self,
        frames_per_second: int,
        begin_note: int,
        classes_num: int,
    ):
        r"""Class for processing MIDI events to targets.

        Args:
          segment_seconds: float, e.g., 10.0
          frames_per_second: int, e.g., 100
          begin_note: int, A0 MIDI note of a piano, e.g., 21
          classes_num: int, e.g., 88
        """
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process2(self,
                 start_time: float,
                 segment_seconds: float,
                 prettymidi_events,
                 note_shift=0):

        segment_frames = int(self.frames_per_second * segment_seconds)

        note_events = []
        # E.g. [
        #   {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
        #   {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        #   ...]

        '''
        for prettymidi_event in prettymidi_events:
            if (start_time < prettymidi_event.start < start_time + self.segment_seconds) or (start_time < prettymidi_event.end < start_time + self.segment_seconds):
                note_events.append({
                    'midi_note': prettymidi_event.pitch,
                    'onset_time': prettymidi_event.start,
                    'offset_time': prettymidi_event.end,
                    'velocity': prettymidi_event.velocity,
                })
        '''

        for prettymidi_event in prettymidi_events:
            if (start_time < prettymidi_event['start'] < start_time + segment_seconds) or (start_time < prettymidi_event['end'] < start_time + segment_seconds):
                note_events.append({
                    'midi_note': prettymidi_event['pitch'],
                    'onset_time': prettymidi_event['start'],
                    'offset_time': prettymidi_event['end']
                })

        # Prepare targets.
        frames_num = int(round(segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))

        mask_roll = np.ones((frames_num, self.classes_num))
        # mask_roll is used for masking out cross segment notes

        # ------ 2. Get note targets ------
        # Process note events to target.
        for note_event in note_events:
            # note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note)
            # There are 88 keys on a piano.

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

#                     velocity_roll[max(bgn_frame, 0) : min(fin_frame, segment_frames) + 1, piano_note] = note_event['velocity']

#                     if fin_frame < segment_frames:
#                         offset_roll[fin_frame, piano_note] = 1

#                         # Vector from the center of a frame to ground truth offset.
#                         reg_offset_roll[fin_frame, piano_note] = (note_event['offset_time'] - start_time) - (
#                             fin_frame / self.frames_per_second
#                         )

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset.
                        reg_onset_roll[bgn_frame, piano_note] = (note_event['onset_time'] - start_time) - (
                            bgn_frame / self.frames_per_second
                        )

                    # Mask out segment notes.
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])

        target_dict = {
            'onset_roll': onset_roll,
            'reg_onset_roll': reg_onset_roll,
            'frame_roll': frame_roll,
            'mask_roll': mask_roll,
        }

        return target_dict, note_events
    
    
    def pkl2roll(self,
                 start_time: float,
                 segment_seconds: float,
                 note_events,
                 note_shift=0):

        segment_frames = int(self.frames_per_second * segment_seconds)

        # E.g. [
        #   {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
        #   {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        #   ...]
        frames_num = int(round(segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))

        # ------ 2. Get note targets ------
        # Process note events to target.
        for note_event in note_events:
            # note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note)
            # There are 88 keys on a piano.

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

#                     velocity_roll[max(bgn_frame, 0) : min(fin_frame, segment_frames) + 1, piano_note] = note_event['velocity']

#                     if fin_frame < segment_frames:
#                         offset_roll[fin_frame, piano_note] = 1

#                         # Vector from the center of a frame to ground truth offset.
#                         reg_offset_roll[fin_frame, piano_note] = (note_event['offset_time'] - start_time) - (
#                             fin_frame / self.frames_per_second
#                         )

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset.
                        reg_onset_roll[bgn_frame, piano_note] = (note_event['onset_time'] - start_time) - (
                            bgn_frame / self.frames_per_second
                        )

                    # Mask out segment notes.
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])

        target_dict = {
            'onset_roll': onset_roll,
            'reg_onset_roll': reg_onset_roll,
            'frame_roll': frame_roll,
            'mask_roll': mask_roll,
        }

        return target_dict    

    def process_beats(self, start_time: float, segment_seconds: float, beats, note_shift=0):

        segment_frames = int(self.frames_per_second * segment_seconds)

        beat_events = []

        for beat_time in beats:
            if start_time < beat_time < start_time + segment_seconds:
                beat_events.append({'beat_time': beat_time})

        # Prepare targets.
        frames_num = int(round(segment_seconds * self.frames_per_second)) + 1
        beat_roll = np.zeros(frames_num)
        reg_beat_roll = np.ones(frames_num)
        
        # ------ 2. Get note targets ------
        # Process note events to target.
        for beat_event in beat_events:
            # note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}

            bgn_frame = int(round((beat_event['beat_time'] - start_time) * self.frames_per_second))

            beat_roll[bgn_frame] = 1

            # Vector from the center of a frame to ground truth onset.
            reg_beat_roll[bgn_frame] = (beat_event['beat_time'] - start_time) - (
                bgn_frame / self.frames_per_second
            )

        reg_beat_roll = self.get_regression(reg_beat_roll)

        target_dict = {
            'beat_roll': beat_roll,
            'reg_beat_roll': reg_beat_roll,
        }

        return target_dict, beat_events

    def extend_pedal(self, note_events, pedal_events):
        r"""Update the offset of all notes until pedal is released.

        Args:
            note_events: list of dict, e.g., [
                {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
                {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
                ...]
            pedal_events: list of dict, e.g., [
                {'onset_time': 696.46875, 'offset_time': 696.62604},
                {'onset_time': 696.8063, 'offset_time': 698.50836},
                ...]

        Returns:
            ex_note_events: list of dict, e.g., [
                {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
                {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
                ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0  # index of note events
        while pedal_events:  # Go through all pedal events.
            pedal_event = pedal_events.popleft()
            buffer_dict = {}  # keys: midi notes; value of each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal,
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:

                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        # Multiple same note inside a pedal.
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset.
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx

                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal.
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            # Append left notes.
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        r"""Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by
        Regressing Onsets and Offsets Times, 2020.

        Args:
            input: (frames_num,)

        Returns:
            output: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9,
                0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1.0 / self.frames_per_second
        output = np.ones_like(input)

        locts = np.where(input < 0.5)[0]
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i + 1]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0.0, 0.05) * 20
        output = 1.0 - output

        return output    