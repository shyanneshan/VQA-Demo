# use scispaCy for medical ner

import scispacy
import spacy
import en_ner_bc5cdr_md


def ner(text):
    nlp = en_ner_bc5cdr_md.load()
    doc = nlp(text)
    res = []
    for entity in doc.ents:
        res.append([str(entity), str(entity.label_)])
    return res


