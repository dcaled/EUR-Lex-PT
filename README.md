# EUR-Lex-PT

## Classifier

**Installation**
1. Please, install the packages listed on the ```requirements.txt``` file. 

    1.1 Run ```virtualenv -p python3 envname```

    1.2 Run ```envname/bin/pip install -r requirements.txt```

2. Download CBOW 300 dimensions (cbow_s300.zip) from [NILC-Embeddings](http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/cbow_s300.zip) and unzip it to ```data/embeddings/``` directory.

3. Download the pre-processed EUR-Lex-PT dataset ```clean_txt_withallwords.json``` from [here]() to ```data/``` directory.

4. Run ```classifier/hierarchical.py```.


## Data

### EUR-Lex-PT

The pre-processed EUR-Lex-PT is formated with a JSON representation as follows: 

```
{   
    "celex": "eur-lex_identifier", 
    "url": "eur-lex_url",
    "eurovoc_ids": ["descriptor_index_1", "descriptor_index_2", ..., "descriptor_index_n"], 
    "eurovoc_mts": ["micro-thesaurus_index_1", "micro-thesaurus_index_2", ..., "micro-thesaurus_index_n"],
    "eurovoc_dom": ["domain_index_1", "domain_index_2", ..., "domain_index_n"], 
    "txt": "document_text"
}
```

The documents' sentences (```"txt"``` field) are separated by a ```__SENT__``` delimiter.



### EUR-Lex-PT LIBSVM format

The EUR-Lex-PT corpus is also available under the LIBSVM format at [EUR-Lex-PT LIBSVM](https://drive.google.com/drive/folders/1QE9ICV0D-qK9EprVxAMxWI9nuLpfoxXI?usp=sharing): 

- Each LIBSVM file contains a header in the format:

```<#instances> <#features> <#classes>```

- The other lines of the LIBSVM files are in the format:

```<label> <index1>:<value1> <index2>:<value2> ...```


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
