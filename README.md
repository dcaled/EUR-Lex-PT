# EUR-Lex-PT

## Classifier

1. Please, install the packages listed on the ```requirements.txt``` file.
2. Run ```classifier/hierarchical.py```.


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
