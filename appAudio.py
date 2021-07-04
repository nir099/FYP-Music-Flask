import numpy as np
import tensorflow as tf
import librosa as lbr
from flask import Flask
import boto3
import audioProcess
from flask_cors import CORS
from flask import request
from mido import MidiFile

import pretty_midi as pm
from   librosa.display     import specshow

from keras.models          import load_model
from keras.utils.vis_utils import model_to_dot

import noisereduce as nr

app = Flask(__name__)
CORS(app, max_age="21000", send_wildcard="True")
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
audioTable = dynamodb.Table('Audio-zo3cmm6a6fblhpk6g4trt53aju-dev')
bucketName = 'fypbucket105550-dev'
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr

melsMinMin, melsMinMax, melsMeanMin, melsMeanMax, melsMaxMin, melsMaxMax = -43, -36, -26, -3, 37, 44

onMod, offMod, actMod, volMod = map(lambda name: load_model('Magenta {}.hdf5'.format(name), compile=False), ['Onsets 68.96', 'Offsets 47.02', 'Frame 80.99', 'Velocity 94.28'])


#####
def processAudio(userAudioClip):
    rate, songName, noiceName = 16_000, userAudioClip, 'noiceNew.ogg'
    data = lbr.effects.trim(lbr.load(songName, rate)[0])[0]

    # select section of data that is noise
    noisy_part = lbr.effects.trim(lbr.load(noiceName, rate)[0])[0]

    # perform noise reduction
    song = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
    songLen = int(lbr.get_duration(song, rate))

    mels = lbr.power_to_db(lbr.magphase(lbr.feature.melspectrogram(song, rate, n_mels=229, fmin=30, htk=True))[0])


    nFrames = lbr.time_to_frames(5, rate) + 1
    melsReshaped = np.pad(mels, [(0, 0), (0, -mels.shape[1] % nFrames)], 'minimum').T.reshape((-1, nFrames, len(mels)))

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
    frames = onsets | (actProb > .8).astype(np.int8) # Ensure that any frame with an onset prediction is considered active.

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
    notes = midi.get_pitch_class_histogram()
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
    return True


def upload_audio_Midi_file(file_name,userId, audioId, object_name, link):

    # Upload the file
    try:
        response = s3.upload_file(file_name, bucketName, object_name)
        response = s3.upload_file('audioMidiHTML.html', bucketName, link, ExtraArgs={'ContentType': "text/html"})
        audioTable.update_item(
            Key={
                'id': audioId,
            },
            UpdateExpression='SET midiLink = :val1',
            ExpressionAttributeValues={
                ':val1': userId + '/audioMidi/'+ audioId + '/' + file_name
            }
        )

        audioTable.update_item(
            Key={
                'id': audioId,
            },
            UpdateExpression='SET audioHtmlFile = :val1',
            ExpressionAttributeValues={
                ':val1': userId + '/audioMidiHTML/'+ audioId + '/' + 'audioMidiHTML.html'
            }
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_compare_file(file_name,userId, audioId, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    try:
        response = s3.upload_file(file_name, bucketName, object_name)
        audioTable.update_item(
            Key={
                'id': audioId,
            },
            UpdateExpression='SET comparedFile = :val1',
            ExpressionAttributeValues={
                ':val1': userId + '/comparedFile/'+ audioId + '/' + file_name
            }
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True


def ReportWrite(output):
    Report=open("Report.txt","w")
    if (len(output)!=1):
        Report.write("Audio Sheet Result")
        Report.write("\n")
        index = 1
        for q in output:
            Report.write(str(index) + ' ' + q)
            index += 1
            Report.write("\n")
    else:
       for q in output:
            Report.write(q)
    Report.close()
    return True

def CompareMidi(t1,t2):
    mid1 = MidiFile(t2, clip=True)
    mid2 = MidiFile(t1, clip=True)

    output=[]
    track1=[]
    track2=[]

    for msg in mid1.tracks[1]:
        if (msg.type=='note_on' and (int(str(msg).split(' ')[3].split('=')[1]) > 0 and int(str(msg).split(' ')[4].split('=')[1]) >= 0)):
           track1.append([msg.note,msg.time])
    for msg in mid2.tracks[1]:
        if (msg.type=='note_off'):
           track2.append([msg.note,msg.time])

    no1=len(track1)
    no2=len(track2)
    lenrun = no1
    if(no2 < no1 ):
      lenrun = no2
    for i in range(0,lenrun):
        if(track1[i][0]==track2[i][0]):
            output.append(lbr.midi_to_note(track1[i][0])+"     "+lbr.midi_to_note(track2[i][0])+"     Correct")
            #output.append(chr(track1[i][0])+" "+str(track1[i][1])+" "+chr(track2[i][0])+" "+str(track2[i][1])+" Correct")

        else:
            output.append(lbr.midi_to_note(track1[i][0])+"     "+lbr.midi_to_note(track2[i][0])+"     Wrong")
            #output.append(chr(track1[i][0])+" "+str(track1[i][1])+" "+chr(track2[i][0])+" "+str(track2[i][1])+" Wrong")

    return output

def createAudioHTML(link):
    templateAudio = """
<!-- Midi HTML -->
<section id="sectionAudio">
  <midi-visualizer
    type="waterfall"
    src="{midiAudioLink}"
  ></midi-visualizer>
  <midi-player
    src="{midiAudioLink}"
    sound-font
    visualizer="#sectionAudio midi-visualizer"
  ></midi-player>
</section>

<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.21.0/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.1.1"></script>
"""
    contextAudio = {
        "midiAudioLink": 'https://' + bucketName + '.s3.ap-south-1.amazonaws.com/' + link,
    }
    with  open('audioMidiHTML.html','w') as myfile:
        myfile.write(templateAudio.format(**contextAudio))
    return True

@app.route('/')
def index():
    return "This is the main page."

@app.route("/ping", methods=['GET'])
def test():
    return 'This is a ping'

@app.route("/api/audio/<userId>/<audioId>", methods=['GET'])
def audioClip(userId, audioId):
    selectedAudioData = audioTable.get_item(
        Key={
            'id': audioId
        }
    )
    selectedAudioData = selectedAudioData['Item']
    s3.download_file(bucketName, 'public/' + selectedAudioData['audioLink'], 'currentAudioFile' )
    processAudio('currentAudioFile')
    link = 'public/' + userId + '/audioMidiHTML/'+ audioId + '/' +'audioMidiHTML.html'
    htmlLink = 'public/' + userId + '/audioMidi/'+ audioId + '/' +'processedAudioMidiClip.mid'
    createAudioHTML(htmlLink)
    success = upload_audio_Midi_file('processedAudioMidiClip.mid', userId, audioId,  'public/' + userId + '/audioMidi/'+ audioId + '/' +'processedAudioMidiClip.mid', link )
    return 'ok'

@app.route("/api/compare/<userId>", methods=['POST'])
def compare(userId):
    data = request.get_json()
    audioMidi = str(data.get('audioMidi'))
    sheetMidi = str(data.get('sheetMidi'))
    audioId = str(data.get('audioId'))
    s3.download_file(bucketName,  'public/' +    audioMidi , 'compareAudioMidi.mid');
    s3.download_file(bucketName, 'public/' + sheetMidi , 'compareSheetMidi.mid')
    comparedOutput = CompareMidi('compareSheetMidi.mid' , 'compareAudioMidi.mid')
    ReportWrite(comparedOutput)
    upload_compare_file('Report.txt', userId, audioId, 'public/' + userId + '/comparedFile/' + audioId + '/' +  'Report.txt')
    return 'ok'

if __name__ == '__main__':
    app.run()