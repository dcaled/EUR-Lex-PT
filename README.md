# EUR-Lex-PT

## Classifier

**Installation**
1. Please, install the packages listed on the ```requirements.txt``` file. 

1.1 Run ```virtualenv -p python3 envname```

1.2 Run ```envname/bin/pip install -r requirements.txt```

2. Download CBOW 300 dimensions (cbow_s300.zip) from [NILC-Embeddings](143.107.183.175:22980/download.php?file=embeddings/word2vec/cbow_s300.zip) and unzip it to ```data/embeddings/``` directory.

3. Run ```classifier/hierarchical.py```.


## Data

### EuroVoc Thesaurus

The EuroVoc thesaurus can be found at the ```data/eurovoc.json``` file. Each descriptor is organized according to the structure presented bellow. As the descriptors may contain equivalent terms (preferred and non-preferred), we list all these terms under the ```terms``` list.

```
"descriptor_index": {
        "id": "descriptor_index",
        "terms": [
            "descriptor_label_1",
            "descriptor_label_2",
        ],
        "pt": "descriptor_label",
        "mts": {
            "micro-thesaurus_index_1": "micro-thesaurus_label_1",
            "micro-thesaurus_index_2": "micro-thesaurus_label_2"
        },
        "domains": {
            "domain_index_1": "domain_index_1",
            "domain_index_2": "domain_index_2",
        }
    }
 ```
