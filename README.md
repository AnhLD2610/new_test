# NLP-Project
CS224N Final Project

##### Team members*
- Jon Deaton (jdeaton@stanford.edu)
- Austin Jacobs (ajacobs7@stanford.edu)
- Kathleen Kenealy (kkenealy@stanford.edu)

\* Equal contributions

### Setup
1. Download the CNN-DailyMail data set by following these [instructions](https://github.com/abisee/cnn-dailymail). 
    - Make sure that your data set is saved in a directory
    with the following directory structure
    
    
    CNN-DailyMail  
    ├── cnn  
    │   └── stories  
    ├── dailymail  
    │   └── stories  
    └── preprocessed  
        ├── cnn_stories_tokenized  
        ├── dm_stories_tokenized  
        └── finished_files  
        
2. Download NYT Annotated Dataset at the following link. The dataset is only available to those who are affiliated with Stanford University.
    - https://drive.google.com/drive/folders/1BkwMlkUiii4YPjB51g9LuQdLzwg7IqMx?usp=sharing
    - See the following links for more information about the dataset
        - LDC: https://catalog.ldc.upenn.edu/LDC2008T19
        - Manual: https://catalog.ldc.upenn.edu/docs/LDC2008T19/new_york_times_annotated_corpus.pdf
3. Download GloVe embeddings.
4. Create a file `config/config.ini` with the following format. 
```
[Data]
cnn_path = path/to/cnn/dataset
nyt_path = path/to/nyt/dataset
emb_path = path/to/GloVe/embeddings

[Cuda]
use = True/False
```

5. Run `python -m NYTDataset.preprocess_nyt` in the main directory. Depending on you're computer, this should take about 35-40 minutes to run.
6. Install requirements


    python setup.py install 


## Training
To train a model run 


    python -m training.run
