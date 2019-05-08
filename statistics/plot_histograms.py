import io
import numpy as np
import matplotlib.pyplot as plt
import operator
import json


def labels_per_doc(celex_labels):
    x_dom, x_mts, x_terms = [], [], []
    for celex in celex_labels.keys():
        x_dom += [celex_labels[celex]['dom']]
        x_mts += [celex_labels[celex]['mts']]
        x_terms += [celex_labels[celex]['terms']]

    bins = np.linspace(0, 15, 15)

    plt.figure(0, figsize=(9, 3.5))
    plt.style.use('seaborn-deep')
    plt.hist([x_dom, x_mts, x_terms], bins, label=[
             'domains', 'micro-thesauri', 'descriptors'])
    plt.legend(loc='upper right')
    #plt.title('Labels per document')
    plt.xlabel("Labels per document")
    plt.ylabel("Number of documents")
    plt.savefig("labels_per_instance_color.png")

    plt.figure(1, figsize=(9, 4))
    plt.style.use('grayscale')
    plt.hist([x_dom, x_mts, x_terms], bins, label=[
             'domains', 'micro-thesauri', 'descriptors'])
    plt.legend(loc='upper right')
    #plt.title('Labels per document')
    plt.xlabel("Labels per document")
    plt.ylabel("Number of documents")
    plt.savefig("labels_per_instance_gray.png")


def docs_per_label(docs_per_lbl):
    plt.style.use('seaborn-deep')

    plt.figure(0, figsize=(9, 3.5))

    x_dom = sorted(docs_per_lbl['doms'],
                   key=docs_per_lbl['doms'].get, reverse=True)
    y_dom = sorted(list(docs_per_lbl['doms'].values()), reverse=True)

    x_mts = sorted(docs_per_lbl['mts'],
                   key=docs_per_lbl['mts'].get, reverse=True)
    y_mts = sorted(list(docs_per_lbl['mts'].values()), reverse=True)

    plt.subplot(1, 2, 1)
    plt.ylim(0, 100000)
    plt.bar(x_dom, y_dom)
    plt.xticks([])
    plt.title('Level: domain')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")

    #plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.ylim(0, 100000)
    plt.xticks([])
    plt.yticks([])
    plt.bar(x_mts, y_mts)
    plt.title('Level: micro-thesaurus')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")

    plt.savefig("docs_per_lbl_color.png")


def contains_portugal(kw_entry):
    terms = kw_entry['terms']
    micro_thesauri = kw_entry['micro-thesauri']
    domains = kw_entry['domains']

    for t in terms:
        if 'portugal' in t.lower():
            return True

    for code in micro_thesauri:
        if 'portugal' in micro_thesauri[code].lower():
            return True

    for code in domains:
        if 'portugal' in domains[code].lower():
            return True

    return False

def get_portuguese_kw_map():
      ######## Portugal-related statistics ########

      eurovoc_path = '../data/eurovoc.json'

      kw_map = {}
      # https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
      with open(eurovoc_path, 'r', encoding="utf8") as f:
            kw_map = json.load(f)

      # Filter by relevant portuguese keywords.
      filtered_kw_map = {}
      for kw in kw_map:    
            if contains_portugal(kw_map[kw]):
                  #print(kw_map[kw])
                  filtered_kw_map[kw] = kw_map[kw]

      return filtered_kw_map 

# MAIN
celex_labels = {}
docs_per_lbl = {'doms': {}, 'mts': {}, 'terms': {}}
# insert here the path to stratification files.
stratification_path = '../data/stratification'


filtered_kw_map = get_portuguese_kw_map()

splits = ['train', 'val', 'test']
for split in splits:
    with io.open('{}/{}.txt'.format(stratification_path, split)) as f:
        for i, line in enumerate(f):

            celex, doms, mts, terms = line.split('|')
            doms = doms.strip().split(',')
            mts = mts.strip().split(',')
            terms = terms.strip().split(',')

            # Each celex corresponds to a single document and has an associated (doms, mts, terms)
            celex_labels[celex] = {
                'dom': len(doms),
                'mts': len(mts),
                'terms': len(terms)
            }

            # TODO: check current (doms, mts, terms) codes and if any of them is found in filtered_kw_map, store them in filtered_celex_labels.
            filtered_celex_labels = {}

            for lbl in doms:
                docs_per_lbl['doms'][lbl] = docs_per_lbl['doms'].get(
                    lbl, 0) + 1
            for lbl in mts:
                docs_per_lbl['mts'][lbl] = docs_per_lbl['mts'].get(lbl, 0) + 1
            for lbl in terms:
                docs_per_lbl['terms'][lbl] = docs_per_lbl['terms'].get(
                    lbl, 0) + 1


######## Labels per doc ########
labels_per_doc(celex_labels)

######## Docs per label ########
docs_per_label(docs_per_lbl)






