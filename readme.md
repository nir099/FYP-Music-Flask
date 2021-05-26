from google.colab import drive
drive.mount('/content/gdrive')
%cd gdrive/My Drive/project/tf-end-to-end/tf-end-to-end 
#################### change
print('connected to my drive')

! git clone https://github.com/nir099/fyp-omr
print('done')

%cd fyp-omr

%tensorflow_version 1.x

#################### change data set path ( corpus )
!python ctc_training.py -corpus DataSet/DataSet  -vocabulary Data/vocabulary_semantic.txt  -save_model models/trained_semantic_model


#################### change model path ( model )
!python ctc_predict.py -image Data/Example/sheet.jpg -model ModelS/semantic_model.meta -vocabulary Data/vocabulary_semantic.txt

############# with single
python omr_predict.py -image Data/Example/mary.jpg -model Semantic-Model/trained_semantic_model-3000.meta -vocabulary Data/vocabulary_semantic.txt -single false