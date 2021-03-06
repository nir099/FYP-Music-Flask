import omr_utils
import cv2
import numpy as np
import tensorflow as tf
from midi.player import *
from midiutil.MidiFile import MIDIFile
from datetime import datetime

from crop_image import default

# %config InlineBackend.figure_format = 'retina'

# import pretty_midi as pm
# import librosa     as lbr
# from   librosa.display     import specshow

# from keras.models          import load_model
# from keras.utils.vis_utils import model_to_dot

################## flask

from flask import Flask
from flask_cors import CORS
import boto3

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open('Data/vocabulary_semantic.txt','r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf.train.import_meta_graph('Semantic-Model/trained_semantic_model-3000.meta')
saver.restore(sess,'Semantic-Model/trained_semantic_model-3000.meta'[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

app = Flask(__name__)
CORS(app)
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sheetMusicTable = dynamodb.Table('Sheet-zo3cmm6a6fblhpk6g4trt53aju-dev')
audioTable = dynamodb.Table('')
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr


def upload_sheetMusic_Midi_file(file_name, bucket,userId, imageId, sheetMusicId, object_name, link):
    # Upload the file
    try:
        response = s3.upload_file(file_name, bucket, object_name)
        response = s3.upload_file('sheetMidiHTML.html', bucket, link, ExtraArgs={'ContentType': "text/html"})
        sheetMusicTable.update_item(
            Key={
                'id': sheetMusicId,
            },
            UpdateExpression='SET midiLink = :val1',
            ExpressionAttributeValues={
                ':val1': userId + '/sheetMidi/'+ sheetMusicId + '/' + imageId
            }
        )
        sheetMusicTable.update_item(
            Key={
                'id': sheetMusicId,
            },
            UpdateExpression='SET midiHtmlFile = :val1',
            ExpressionAttributeValues={
                ':val1': userId + '/sheetMidiHTML/'+ sheetMusicId + '/' + 'sheetMidiHTML.html'
            }
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True

def createSheetHTML(link):
    templateSheet = """
<!-- Midi HTML -->
<section id="sectionSheet">
  <midi-visualizer
    type="waterfall"
    src="{midiSheetLink}"
  ></midi-visualizer>
  <midi-player
    src="{midiSheetLink}"
    sound-font
    visualizer="#sectionSheet midi-visualizer"
  ></midi-player>
</section>

<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.21.0/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.1.1"></script>
    """
    contextSheet = {
        "midiSheetLink": 'https://fypbucket105550-dev.s3.ap-south-1.amazonaws.com/' + link,
    }
    with  open('sheetMidiHTML.html','w') as myfile:
        myfile.write(templateSheet.format(**contextSheet))

@app.route('/')
def index():
    return "This is the main page."


@app.route("/api/sheetMusic/<userId>/<sheetMusicId>", methods=['GET'])
def sheetMusic(userId, sheetMusicId):
    midix = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    volume = 100
    selectedSheetMusicData = sheetMusicTable.get_item(
        Key={
            'id': sheetMusicId
        }
    )

    selectedSheetMusicData = selectedSheetMusicData['Item']
    s3.download_file('fypbucket105550-dev', 'public/' + selectedSheetMusicData['sheetLink'], 'currentImageFile' )
    midix.addTrackName(track, time, "Track")
    midix.addTempo(track, time, 60)

    # image setup
    image_set = default('currentImageFile')
    word_set = []
    for i in range(len(image_set)):
        image = image_set[i]
        image = omr_utils.resize(image, HEIGHT)
        image = omr_utils.normalize(image)
        image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

        seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

        prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })

        str_predictions = omr_utils.sparse_tensor_to_strs(prediction)
        SEMANTIC = ''
        for w in str_predictions[0]:
            SEMANTIC += int2word[w] + '\n'
        print(SEMANTIC)
        word_set.append(SEMANTIC)

    # gets the audio file
    for i in range(len(word_set)):
        audio, duration, freq, notes = get_sinewave_audio(word_set[i])
        for i in range(len(notes)):
            if notes[i] != 'rest':
                print(notes[i] , freq[i])
                midix.addNote(track, channel, freq[i], time, duration[i], volume)
            time +=  duration[i]

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y%H-%M-%S")
    binfile = open("processedAudio" + ".mid", 'wb')
    midix.writeFile(binfile)
    link = 'public/' + userId + '/sheetMidiHTML/'+ sheetMusicId +'/' + 'sheetMidiHTML.html'
    htmlLink = 'public/' + userId + '/sheetMidi/'+ sheetMusicId +'/' +'processedAudio.mid'
    createSheetHTML(htmlLink)
    binfile.close()
    success = upload_sheetMusic_Midi_file('processedAudio.mid', 'fypbucket105550-dev', userId, 'processedAudio.mid', sheetMusicId,  'public/' + userId + '/sheetMidi/'+ sheetMusicId +'/' +'processedAudio.mid', link )
    return 'ok'

if __name__ == '__main__':
    app.run()