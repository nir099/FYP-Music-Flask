from IPython.display import display, SVG, Audio
from matplotlib      import pyplot as plt
%config InlineBackend.figure_format = 'retina'

import numpy       as np
import pretty_midi as pm
import librosa     as lbr
from   librosa.display     import specshow

from keras.models          import load_model
from keras.utils.vis_utils import model_to_dot

import noisereduce as nr
# load data

def processAudio(userAudioClip):
    rate, songName, noiceName = 16_000, userAudioClip, 'noiceNew.ogg'
    data = lbr.effects.trim(lbr.load(songName, rate)[0])[0]

    # select section of data that is noise
    noisy_part = lbr.effects.trim(lbr.load(noiceName, rate)[0])[0]

    # perform noise reduction
    song = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
    songLen = int(lbr.get_duration(song, rate))
    melsMinMin, melsMinMax, melsMeanMin, melsMeanMax, melsMaxMin, melsMaxMax = -43, -36, -26, -3, 37, 44

    mels = lbr.power_to_db(lbr.magphase(lbr.feature.melspectrogram(song, rate, n_mels=229, fmin=30, htk=True))[0])


    nFrames = lbr.time_to_frames(5, rate) + 1
    melsReshaped = np.pad(mels, [(0, 0), (0, -mels.shape[1] % nFrames)], 'minimum').T.reshape((-1, nFrames, len(mels)))

    onMod, offMod, actMod, volMod = map(lambda name: load_model('Magenta {}.hdf5'.format(name), compile=False),
                                        ['Onsets 68.96', 'Offsets 47.02', 'Frame 80.99', 'Velocity 94.28'])

    for model in [onMod, offMod, actMod, volMod]: display(SVG(model_to_dot(model, True, False, 'LR').create(format='svg')))

    onProb, offProb, volProb = map(lambda mod: mod.predict(melsReshaped, 32, 1), [onMod, offMod, volMod])

    actProb = actMod.predict([onProb, melsReshaped, offProb], 32, 1)


    onProb, actProb, volProb = map(lambda arr: np.vstack(arr)[:mels.shape[1]], [onProb, actProb, volProb])

    midi = pm.PrettyMIDI(initial_tempo=lbr.beat.tempo(song, rate).mean())
    midi.lyrics += [pm.Lyric('Automatically transcribed from audio:\r\n\t' + songName, 0),
                    pm.Lyric('Used software created by Boris Shakhovsky', 0)]
    track = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'), name='Acoustic Grand Piano')
    midi.instruments += [track]

    intervals, frameLenSecs = {}, lbr.frames_to_time(1, rate) # Time is in absolute seconds, not relative MIDI ticks
    onsets = (onProb > .95).astype(np.int8)
    frames = onsets | (actProb > .5).astype(np.int8) # Ensure that any frame with an onset prediction is considered active.

    def EndPitch(pitch, endFrame):
        track.notes += [pm.Note(int(volProb[intervals[pitch], pitch] * 80 + 10), pitch + 21,
                                intervals[pitch] * frameLenSecs, endFrame * frameLenSecs)]
        del intervals[pitch]

    # Add silent frame at the end so we can do a final loop and terminate any notes that are still active:
    for i, frame in enumerate(np.vstack([frames, np.zeros(frames.shape[1])])):
        for pitch, active in enumerate(frame):
            if active:
                if pitch not in intervals:
                    if onsets is None: intervals[pitch] = i
                    elif onsets[i, pitch]: intervals[pitch] = i # Start a note only if we have predicted an onset
                    #else: Even though the frame is active, there is no onset, so ignore it
                elif onsets is not None:
                    if (onsets[i, pitch] and not onsets[i - 1, pitch]):
                        EndPitch(pitch, i)   # Pitch is already active, but because of a new onset, we should end the note
                        intervals[pitch] = i # and start a new one
            elif pitch in intervals: EndPitch(pitch, i)

    if track.notes: assert len(frames) * frameLenSecs >= track.notes[-1].end, 'Wrong MIDI sequence duration'

    gamma = [n for _, n in sorted([(count, ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][i])
                                for i, count in enumerate(notes)], reverse=True)[:7]]
    blacks = sorted(n for n in gamma if len(n) > 1)


    chroma = lbr.feature.chroma_cqt(song, rate).sum(1)
    major = [np.corrcoef(chroma, np.roll([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], i))[0, 1] for i in range(12)]
    minor = [np.corrcoef(chroma, np.roll([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], i))[0, 1] for i in range(12)]
    keySignature = (['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][
        major.index(max(major)) if max(major) > max(minor) else minor.index(max(minor)) - 3]
                    + ('m' if max(major) < max(minor) else ''))

    midi.key_signature_changes += [pm.KeySignature(pm.key_name_to_key_number(keySignature), 0)]
    midi.write('processedAudioMidiClip.mid')
    return