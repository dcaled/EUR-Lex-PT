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


def docs_per_label(docs_per_lbl, suffix):
    
    plt.clf()

    plt.style.use('seaborn-deep')

    plt.figure(0, figsize=(9, 3.5))

    x_dom = sorted(docs_per_lbl['doms'],
                   key=docs_per_lbl['doms'].get, reverse=True)
    y_dom = sorted(list(docs_per_lbl['doms'].values()), reverse=True)

    x_mts = sorted(docs_per_lbl['mts'],
                   key=docs_per_lbl['mts'].get, reverse=True)
    y_mts = sorted(list(docs_per_lbl['mts'].values()), reverse=True)

    print("> x_dom:\t{}".format(x_dom))

    print("> x_mts:\t{}".format(x_mts))


    plt.subplot(1, 2, 1)
#    plt.ylim(0, 100000)

    yy_max = int(max(y_dom) * 1.15)

    plt.ylim(0, yy_max)

    plt.bar(x_dom, y_dom)
    plt.xticks([])
    plt.title('Level: domain')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")

    #plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.ylim(0, yy_max)
    #plt.ylim(0, 100000)
    plt.xticks([])
    plt.yticks([])
    plt.bar(x_mts, y_mts)
    plt.title('Level: micro-thesaurus')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")

    plt.savefig("docs_per_lbl_color_" + suffix + ".png")

    print('\n')


def contains_target_word(kw_entry, target_word):
    terms = kw_entry['terms']
    micro_thesauri = kw_entry['micro-thesauri']
    domains = kw_entry['domains']




    for t in terms:
        if target_word in t.lower():
            return True

    for code in micro_thesauri:
        if target_word in micro_thesauri[code].lower():
            return True

    for code in domains:
        if target_word in domains[code].lower():
            return True

    return False

def get_portuguese_strings(filtered_kw_entry):
    terms = filtered_kw_entry['terms']
    micro_thesauri = filtered_kw_entry['micro-thesauri']
    domains = filtered_kw_entry['domains']

    filtered_doms = []
    filtered_mts = []
    filtered_terms = []


    for t in terms:
        filtered_terms.append((filtered_kw_entry['id'], t.lower()))

    for code in micro_thesauri:
        filtered_mts.append((code, micro_thesauri[code].lower()))

    for code in domains:
        filtered_doms.append((code, domains[code].lower()))

    return filtered_doms, filtered_mts, filtered_terms

def get_portuguese_kw_map(target_words):
      ######## Portugal-related statistics ########

      eurovoc_path = '../data/eurovoc.json'

      kw_map = {}
      # https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
      with open(eurovoc_path, 'r', encoding="utf8") as f:
            kw_map = json.load(f)

      # Filter by relevant portuguese keywords.
      filtered_kw_map = {}
      filtered_doms = {}
      filtered_mts = {}
      filtered_terms = {}

      for tw in target_words:
            tw_filtered_doms = []
            tw_filtered_mts = []
            tw_filtered_terms = []
            for kw in kw_map:
                  if contains_target_word(kw_map[kw], tw):
                        #print(kw_map[kw])
                        filtered_kw_map[kw] = kw_map[kw]

                        curr_filtered_doms, curr_filtered_mts, curr_filtered_terms = get_portuguese_strings(filtered_kw_map[kw])
                        tw_filtered_doms = tw_filtered_doms + curr_filtered_doms
                        tw_filtered_mts = tw_filtered_mts + curr_filtered_mts
                        tw_filtered_terms = tw_filtered_terms + curr_filtered_terms

            filtered_doms[tw] = tw_filtered_doms
            filtered_mts[tw] = tw_filtered_mts
            filtered_terms[tw] = tw_filtered_terms

      return filtered_kw_map, filtered_doms, filtered_mts, filtered_terms

# MAIN
celex_labels = {}
docs_per_lbl = {'doms': {}, 'mts': {}, 'terms': {}}
# insert here the path to stratification files.
stratification_path = '../data/stratification'

#target_words = ['portugal']
target_words = ['portugal', 'português', 'portugueses', 'portaria', 'despacho']

filtered_kw_map, filtered_doms, filtered_mts, filtered_terms = get_portuguese_kw_map(target_words)

print('> Filtered terms:')
print('\t{}'.format(filtered_terms))

print('> Filtered doms:')
print('\t{}'.format(filtered_doms))

print('> Filtered mts:')
print('\t{}'.format(filtered_mts))

filtered_celex_labels = {}

#docs_per_pt_lbl = {}
#for k in filtered_kw_map:
#      docs_per_pt_lbl[k] = {'doms': {}, 'mts': {}, 'terms': {}}

#docs_per_portugal_lbl = {'doms': {}, 'mts': {}, 'terms': {}}
docs_per_target_word = {}
for tw in filtered_doms:
      docs_per_target_word[tw] = {'doms': {}, 'mts': {}, 'terms': {}}
for tw in filtered_mts:
      docs_per_target_word[tw] = {'doms': {}, 'mts': {}, 'terms': {}}
for tw in filtered_terms:
      docs_per_target_word[tw] = {'doms': {}, 'mts': {}, 'terms': {}}


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

            for lbl in doms:
                docs_per_lbl['doms'][lbl] = docs_per_lbl['doms'].get(
                    lbl, 0) + 1

                for tw in filtered_doms:
                    for filtered_tuple in filtered_doms[tw]:
                        if lbl == filtered_tuple[0]:
                            docs_per_target_word[tw]['doms'][lbl] = docs_per_target_word[tw]['doms'].get(
                        lbl, 0) + 1

            for lbl in mts:
                docs_per_lbl['mts'][lbl] = docs_per_lbl['mts'].get(lbl, 0) + 1

                for tw in filtered_mts:
                      for filtered_tuple in filtered_mts[tw]:
                        if lbl == filtered_tuple[0]:
                              docs_per_target_word[tw]['mts'][lbl] = docs_per_target_word[tw]['mts'].get(
                        lbl, 0) + 1

            for lbl in terms:
                docs_per_lbl['terms'][lbl] = docs_per_lbl['terms'].get(
                    lbl, 0) + 1

                for tw in filtered_terms:
                  for filtered_tuple in filtered_terms[tw]:
                        if lbl == filtered_tuple[0]:
                              docs_per_target_word[tw]['terms'][lbl] = docs_per_target_word[tw]['terms'].get(
                        lbl, 0) + 1


######## Labels per doc ########
labels_per_doc(celex_labels)



######## Docs per label ########
docs_per_label(docs_per_lbl, 'global')

for current_word in docs_per_target_word:

      print('> keyword: {}'.format(current_word))
      print('{}\n'.format(docs_per_target_word[current_word]))


      if len(docs_per_target_word[current_word]['doms']) > 0 and len(docs_per_target_word[current_word]['mts']) > 0:
            docs_per_label(docs_per_target_word[current_word], current_word)






