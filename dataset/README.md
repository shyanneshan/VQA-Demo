
### the uploaded zip file should be in the required format
```
├── data  
│   └── txt   
│       ├── 1.txt  
│       ├── 2.txt  
│       ├── 3.txt  
│       └── ....txt  
│   └── img  
│       ├── 1.jpg  
│       ├── 2.jpg  
│       ├── 3.jpg  
│       └── ....jpg  
└── └── relationship.csv  
```

### NER model
The models mentioned in the paper is conducted with chinese medical dataset, therefore, the NER models we provided in the system is chinese NER model using BiLSTM-CRF and english NER model using sciSpacy.

### Configuration
You may check and change the default configuration of the path of uploaded file, and generated datasets in ```VQA-Demo/demo/src/main/resources/application.properties``` 


### where to find generated datasets
when the dataset generation part is finished, you may find your dataset in ```VQA-Demo/dataset```


### about labeling module
if you wish to end the processing of labeling one of the datasets, please click "Submit", once you click, the dataset is no longer available for labeling and is heading for the final stage of filling templates.

 
