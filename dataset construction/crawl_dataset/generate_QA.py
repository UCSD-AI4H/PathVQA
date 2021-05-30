############# modify


from __future__ import division, unicode_literals
import nltk
from nltk.parse.stanford import StanfordParser
import requests
from pattern.en import conjugate, lemma, tag
import re
import sys
from Article import Article
import spacy
from nltk.tokenize import sent_tokenize
import logging
from collections import defaultdict
import random
import copy
import question_evaluator
from collections import Counter

import logging
logger = logging.getLogger('AutoML')
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO )


nlp = spacy.load('en')

reload(sys)
sys.setdefaultencoding('utf-8')

VERBOSE = False


APPO = "APPOSITION"
VM = "VERB_MODIFIER"
NV = "NP_VP"
APPOSITION = "NP !< CC !< CONJP < (NP=np1 $.. (/,/ $.. (NP=app $.. /,/)))"
VERB_MODIFIER = "NP=noun > NP $.. VP=modifier"
#NP_VP = "S < (NP=np $.. VP=vp)"
#MAIN_VERB = "S=clause < (VP=main_vp < /VB.?/=tensed !< (VP < /VB.?/))"
#POST_RULE = "S < (NP=np $.. (VP=vp < /VB.?/=tensed))"
NP_VP = "S < (NP=np ?$PP=pp1 $.. (VP=vp < (/VB.?/=tensed ?$.. PP=pp2 ?$.. SBAR=reason)))"
#NP_VP = "S < (NP=np $.. (VP=vp < (/VB.?/=tensed)))"
patterns = [(NV, NP_VP)]
'''
Ignore VBG and VBN, since we have processed it in auxiliary checking.
VBG	Verb, gerund or present participle
VBN	Verb, past participle
add VB to avoid any exceptions
'''
verb_tense_dict = {"VBD": "past", "VBG":"past", "VBN":"past", "VBP": "1sg", "VBZ": "3sg", "VB": "3sg"}
# verb_tense_dict = {"VBD": "3sg", "VBG":"3sg", "VBN":"3sg", "VBP": "1sg", "VBZ": "3sg", "VB": "3sg"}

"""
Remaining work:
- coreference resolution
stanford pos treebank
https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
"""

ques_pool = set()
ques_list = []
answer=[]
answer_no_index=[]
textlist=[]
answer_phrase=[]
noun_plu_set = set()
noun_sing_set = set()

def getTregexResult(text, pattern):
    url = "http://localhost:3456/tregex"
    # request_params = {"pattern": "NP=np < (NP=np1 $.. (/,/ $.. NP=app))"}
    request_params = {"pattern": pattern}
    r = requests.post(url, data=text, params=request_params)
    try:
        js = r.json()
        if js['sentences'][0]:
            # currently, only returns one possible match
            return js['sentences'][0]
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print 'Decoding JSON has failed'
        return None


def getSimpleSentence(text, pattern, TYPE):
    """
    return simplified sentence
    :param text: original sentence
    :param pattern: rule of simplification
    :param TYPE: pattern type
    :return: string
    """

    """index number for captions with numbers before them"""
    global indexnum
    # indexnum = str(0)
    if text[:4].isdigit():
        indexnum=str(text[:4])+'\t'
        text=text[5:]
    elif text[:3].isdigit():
        indexnum=str(text[:3])+'\t'
        text=text[4:]
    elif text[:2].isdigit():
        indexnum = str(text[:2])+'\t'
        text = text[2:]
    elif text[:1].isdigit():
        indexnum = str(text[:1])+'\t'
        text = text[1:]
    js = getTregexResult(text, pattern)
    if not js:
        return None
    # if TYPE == APPO:
    #     main_verb = get_main_verb_tag(text)
    #     return getAppoSimpleSent(js, main_verb)
    # elif TYPE == VM:
    #     main_verb = get_main_verb_tag(text)
    #     return getVerbModSimpleSent(js, main_verb)
    # else:
    ques = []
    for k,v in js.iteritems():
        generateSimpleQues(v,text,indexnum)




def generateSimpleQues(js,text,indexnum):
    if 'namedNodes' in js:
        js = js['namedNodes']
    # parse result
    np, vp, tensed_verb, pp1, pp2, reason = {},{},{},{},{},{}
    res = defaultdict(str)
    for r in js:
        logger.debug('R %s' % r)
        if 'np' in r:
            np = r['np']
        if 'vp' in r:
            vp = r['vp']
        if 'tensed'  in r:
            tensed_verb = r['tensed']
        if 'pp1' in r:
            pp1 = r['pp1']
        if 'pp2' in r:
            pp2 = r['pp2']
        if 'reason' in r:
            reason = r['reason']
    # optional
    if pp1:
        pp1 = get_words_and_tags(pp1)
        pp1 = ' '.join(pp1)
        # first letter should be lower case
        pp1 = pp1[0].lower() + pp1[1:]
    else:
        pp1 = ''
    res['pp1'] = pp1
    # it's possible to handle direct object, but cannot handle indirect object
    inf_pp, inf_pp_left = '',''
    if pp2:
        pp2 = get_words_and_tags(pp2)
        res['pp2'] = pp2
        # # check infinitive, if is, the PP tag should appear before VB.? tag in the vp tree
        # pp_index = vp.find('PP')
        # np_index = vp.find('NP')
        # if pp_index > -1 and pp_index < np_index:
        #     inf_pp_index = pp2.index('\n')
        #     inf_pp = ''
        #     if inf_pp_index > -1:
        #         inf_pp = get_words_and_tags(pp2[:inf_pp_index])
        #         inf_pp = ' '.join(inf_pp)
        #         res['inf_pp'] = inf_pp
        #     inf_pp_left = get_words_and_tags(pp2[inf_pp_index:])
        #     inf_pp_left = ' '.join(inf_pp_left)
        #     res['inf_pp_left'] = inf_pp_left
    if reason:
        reason = get_words_and_tags(reason)
        res['reason'] = reason
    if not np or not vp:
        return

    # get head_verb and its tag
    head_verb, head_verb_tag = get_words_and_tags(tensed_verb, TAG=True)
    if not head_verb or  not head_verb_tag:
        return
    head_verb, head_verb_tag = head_verb[0], head_verb_tag[0]
    np, np_tags = get_words_and_tags(np, TAG=True)
    vp, vp_tags = get_words_and_tags(vp, TAG=True)
    if len(np_tags) < 2:
        return
    head_noun_tag = np_tags[1]
    if head_noun_tag != 'NNP':
        np[0] = np[0].lower()
    # generate binary questions and wh questions
    res['head_verb'] = head_verb
    res['head_verb_tag'] = head_verb_tag
    res['head_noun_tag'] = head_noun_tag
    res['np'] = np
    res['vp'] = vp
    res['vp_tags'] = vp_tags
    res['simpleSent'] = ' '.join([' '.join(np), ' '.join(vp), pp1])
    res['text'] = text
    logger.debug('-'*100)
    logger.debug('SIMPLE SENT %s JSON %s' % (res['simpleSent'], str(js)))
    #generateQues(res)
    generate_question(res,indexnum)



def ques_word_from_subject(np, doc, res):
    """
    Returns question_word, contains pron
    """
    logger.debug('FROM SUBJECT')
    logger.debug('*'*100)
    logger.debug('NP %s' % np)
    logger.debug('DOC %s' % doc)
    logger.debug('NER %s' % ' '.join([ne.text + '\t' + ne.label_ for ne in doc.ents]))
    np_len = len(np)
    has_ner = False
    # if subject contains PRON, do not do coreference resolution, only generates wh question
    subj_pron = False
    qw_res = []
    is_plural = False
    how_many_index = -1
    for word in doc[:np_len]:
        if word.pos_ == 'PRON':
            subj_pron = True
            break
    for index, word in enumerate(doc[:np_len]):
        logger.debug('WORD %s \t TAG %s POS \t %s' % (word, word.tag_, word.pos_))

        "do not need whose question for website captions"
        # if word.tag_.startswith('PRP') and is_possessive(word):
        #     qw_res.append('whose ' + ' '.join(np[index + 1:]))
        #     print  'ne label subject prp', word
        #     answer_phrase.append(word)


        if word.pos_.startswith('NUM') and word.ent_type_ != 'DATE':
            how_many_index = index
            answer_num=word
        if word.ent_type:
            has_ner = True
            answer_ner=word
        if word.tag_.endswith('S'):
            is_plural = True
    # how many question with % fixed
    if how_many_index > -1 and len(np)>how_many_index+1:
        if np[how_many_index+1] == '%':
            qw = 'how many percent '
            if len(np) > how_many_index + 2:
                qw_res.append(qw + ' '.join(np[how_many_index + 2:]))
                answer_phrase.append('answer_how_many_percent' + '\t' + str(answer_num))
        elif np[how_many_index].endswith('%'):
            qw = 'how many percent '
            qw_res.append(qw + ' '.join(np[how_many_index + 1:]))
            answer_phrase.append('answer_how_many_percent' + '\t' +str(answer_num))
        else:
            qw = 'how many ' if is_plural else 'how much '
            qw_res.append(qw + ' '.join(np[how_many_index + 1:]))
            answer_phrase.append('answer_how_many'+'\t' + str(answer_num))
    if has_ner:
        np_doc = nlp(' '.join(np))
        for ne in np_doc.ents:
            logger.debug('ne label %s \t %s ' % (ne.label_, ne.text))
            if not ne.label:
                continue
            # if ne.label_ == 'PERSON':
            #     qw_res.append('who')
                # answer_phrase.append(indexnum + str(word))
    # should be what even if the ner is LOCATION or DATA since this is a NP instead of PP
    # but actually if the ner is GPE, the best question word should be which country/city/state, but it's hard to decide which one to choose
    qw_res.append('what')
    # answer_phrase.append(indexnum +' '.join(np))
    # answer_phrase.append(indexnum + str(word))
    nouns = ' '.join(np)
    if not subj_pron and nouns != '%':
        noun_plu_set.add(nouns.lower()) if is_plural else noun_sing_set.add(nouns.lower())
    return qw_res, subj_pron, answer_phrase


HEAD_WORD_FOR_WHY = ['because', 'since', 'as']
def ques_word_from_object(np, vp, doc, reason, simpleSent, res):
    logger.debug('='*100)
    logger.debug('NP %s' % np)
    logger.debug('VP %s' % vp)
    logger.debug('DOC %s' % doc)
    logger.debug('NER %s' % ' '.join([ne.text + '\t' + ne.label_ for ne in doc.ents]))
    has_why, has_ner = False, False
    verb_break, noun_break = False, False
    plural_noun = False
    verbs = ''
    verbs_no_adp = ''
    first_noun_ending_index = -1

    qw_res = []
    aw_res = []
    # if contains SBAR, then discard this part to avoid weird questions
    if reason and reason[0] in HEAD_WORD_FOR_WHY:
        has_why = True
        vp = vp[:len(vp)-len(reason)]
    np_len = len(np)
    # print('np',np)
    vp_len = len(vp)


    for index, word in enumerate(doc[np_len:np_len+vp_len+1]):

        # whose question
        # if word == 'its' or 'their':
        #     print(doc.replace(word, 'whose'))
        #     ques = post_process(doc)
        #     print(ques)
        #     answer_phrase.append('answer_new_why' + '\t' + word)
        #     ques_list.append(indexnum + 'new_why_question_from_vp' + '\t' + ques.replace(' ' + a, ''))



        logger.debug('WORD %s \t TAG %s POS \t %s NER \t %s' % (word, word.tag_, word.pos_, word.ent_type_))
        if not verb_break:
            if word.pos_ in ['VERB', 'ADP'] and index > 0:
                verbs += word.text + ' '
                if word.pos_ == 'VERB': verbs_no_adp += word.text + ' '
            elif index > 0:
                verb_break = True
        if word.pos_ == 'NOUN':
            if not noun_break: first_noun_ending_index = index
            answer_noun=word
            noun_break = True

        "do need whose question for website captions"
        # if word.tag_.startswith('PRP') and is_possessive(word):
        #     nouns = ''
        #     for word in doc[np_len+index+1:vp_len+1]:
        #         nouns += word.text +' '
        #         if word.pos_ == 'NOUN':
        #             qw_res.append(('whose ' + nouns.strip(), verbs.strip()))
        #             answer_phrase.append(word)


        if word.pos_.startswith('NUM') and word.ent_type_ != 'DATE' and doc[index+1].text != 'hours':
            nouns = ''
            if len(doc) > np_len+index+1 and doc[np_len+index+1].pos_ == 'NOUN': # avoid phrases like one or more fertile females
                for tmpword in doc[np_len+index+1:]:
                    nouns += tmpword.text +' '
                    if tmpword.pos_ == 'NOUN':
                        # add to noun_set
                        nouns = nouns.strip()
                        if tmpword.tag_.endswith('S'):
                            noun_plu_set.add(nouns.lower())
                            plural_noun = True
                        else:
                            noun_sing_set.add(nouns.lower())
                        break
                if nouns.startswith('km/') or nouns.startswith('m/') or nouns.startswith('mph'):
                    qw = 'how fast '
                    ques = simpleSent.replace(word.text, qw, 1)
                    ques = ques.replace(nouns.split(' ')[0],'',1)
                    nouns = ' '.join(nouns.split(' ')[1:])
                    qw_res.append(('how fast ' + nouns.strip(), verbs.strip()))
                    answer_phrase.append('how fast'+'\t' + str(word))
                else:
                    if nouns.startswith('%'):
                        qw = 'how many percent'
                        ques = simpleSent.replace(word.text + ' %', qw, 1)
                        # % is treated as noun, don't generate question like How many percent does the cat eat ?
                        # nouns = nouns.replace('%', '', 1)
                        # qw_res.append(('how many percent ' + nouns.strip(), verbs.strip()))
                    elif word.text.endswith('%'):
                        qw = 'how many percent '
                        ques = simpleSent.replace(word.text, qw, 1)
                    # else:
                    #     qw = 'how many ' if plural_noun else 'how much '
                    #     qw_res.append((qw + nouns.strip(), verbs.strip()))
                    #     # answer_phrase.append('how many/how much'+'\t' + str(word))
                    #     global numberword
                    #     numberword = str(word)
                    #     print('how much question')
                    #     ques = simpleSent.replace(word.text, qw, 1)
                # qw_res.append((False, ques))

        if word.ent_type:
            # print('word', word, word.ent_type, vp, nlp(' '.join(vp)))
            has_ner = True
        # if word.text == 'in' or 'on' or 'within':
        #     qw_res.append(('where', verbs_no_adp.strip()))
    # if contains named entities, generate who/when/where question
    if has_ner:
        vp_doc = nlp(' '.join(vp))
        for ne in vp_doc.ents:
            logger.debug('ne label %s \t %s ' % (ne.label_, ne.text))
            if not ne.label:
                continue
            # if ne.label_ == 'PERSON':
            #     # add to noun_set
            #     noun_sing_set.add(ne.text)
            #     qw_res.append(('who', verbs.strip()))
                # answer_phrase.append('who'+'\t' + str(ne))
            # if ne.label_ in ['LOC', 'GPE']:
            #     qw_res.append(('where', verbs_no_adp.strip())) # should be what since it is a NP instead of PP
                # answer_phrase.append('where'+'\t' + str(ne))
            # if ne.label_ in ['DATE', 'TIME']:
            #     qw_res.append(('when', verbs_no_adp.strip()))
            #     # answer_phrase.append('when'+'\t' + str(ne))
        if has_why:
            qw_res.append(('why', verbs.strip()))
            answer_phrase.append('why'+'\t' + str(ne))

    # always generate what question
    if first_noun_ending_index > 0:
        verbs += ' '.join(vp[first_noun_ending_index+1:])
        qw_res.append(('what', verbs.strip()))
        if len(answer_phrase) < len(ques_list)-1:
           answer_phrase.append('answer_what_for_object'+'\t ' + str(doc[np_len+1:]))

        # print(str(doc))
        # print('str(doc[np_len+1)',str(doc[np_len+1]))
        # print('answer_what_for_object'+'\t ' + str(doc[np_len+1:]))
    for index, word in enumerate(doc[np_len+1:]):
        if word.text == 'due' or word.text == 'because':
            # main_parts =  ques_list.append(' '.join([head_verb, ' '.join(np), verbs, pp1]))
            # main_parts = why_obj_question_part(res, verbs)
            main_parts = binary_question_part(res)
            # print('main',main_parts)
            ques = ' '.join(['why', main_parts])
            # print(ques)
            ques = post_process(ques)
            a = post_process_QA(str(doc[np_len+1+index:]))

            answer_phrase.append('answer_new_why' + '\t' + a)
            # print('due', a)
            ques_list.append(indexnum + 'new_why_question_from_vp' + '\t' + ques.replace(' '+a,''))
            # print('ques', ques.replace(' '+a,''))
            # print('doc',doc[np_len+1+index:])

        if word.text == 'during' or word.text == 'before' or word.text == 'after':
            # main_parts =  ques_list.append(' '.join([head_verb, ' '.join(np), verbs, pp1]))
            # main_parts = why_obj_question_part(res, verbs)
            main_parts = binary_question_part(res)
            # print('main',main_parts)
            ques = ' '.join(['when', main_parts])
            # print(ques)
            ques = post_process(ques)
            a = post_process_QA(str(doc[np_len + 1 + index:]))
            answer_phrase.append('answer_new_when' + '\t' + a)
            # print('when_answer',str(doc[np_len+1+index:]))
            ques_list.append(indexnum + 'new_when_question_from_vp' + '\t' + ques.replace(' '+a,''))
            # print('ques',ques.replace(' '+str(doc[np_len+1+index:]),''))

        if word.text == 'between' or word.text == 'inner' or word.text == 'within':
            # main_parts =  ques_list.append(' '.join([head_verb, ' '.join(np), verbs, pp1]))
            # main_parts = why_obj_question_part(res, verbs)
            main_parts = binary_question_part(res)
            # print('main',main_parts)
            ques = ' '.join(['where', main_parts])
            # print(ques)
            ques = post_process(ques)
            a = post_process_QA(str(doc[np_len + 1 + index:]))
            answer_phrase.append('answer_new_where' + '\t' + a)
            # print('where_answer',str(doc[np_len+1+index:]))
            ques_list.append(indexnum + 'new_where_question_from_vp' + '\t' + ques.replace(' '+a,''))
            # print('ques_where',ques.replace(' '+str(doc[np_len+1+index:]),''))


        if word.pos_.startswith('NUM') and word.ent_type_ != 'DATE':
            main_parts = binary_question_part(res)
            # print('main_wordnum',word.pos_)

            try:
                if doc[np_len + 2+ index].pos_ == 'NOUN' and doc[np_len + 2+ index].text != 'hours':
                   ques = ' '.join([('how many ' if is_plural else 'how much ') + str(doc[np_len + 2+ index].text), main_parts.replace(doc[np_len + 1+ index:np_len + 3+ index].text+' ', '')])
                   # print(ques)
                   ques = post_process(ques)
                   a = word.text
                   # a = post_process_QA(str(doc[np_len + 1 + index:]))
                   answer_phrase.append('answer_new_how_many' + '\t' + a)
                   # print('how_many_answer', a)
                   ques_list.append(indexnum + 'new_how_many_question_from_vp' + '\t' + ques.replace(' ' + a, ''))
                   # print('how many ques', ques)
                else:
                    pass
                   # ques = ' '.join([('how many ' if is_plural else 'how much '),
                   #                   main_parts.replace(doc[np_len + 1 + index].text + ' ', '')])
            except:
                pass
                   # ques = ' '.join([('how many ' if is_plural else 'how much ') , main_parts.replace(doc[np_len + 1 + index].text + ' ', '')])



        # print('tag', word.tag_)

        # if word.tag_ == 'JJ' and word.text != 'present':
        #     main_parts = binary_question_part(res)
        #     ques = ' '.join(['how', main_parts])
        #     ques = post_process(ques)
        #
        #     answer_phrase.append('answer_new_how' + '\t' + word.text)
        #     print('how_answer',word.text)
        #     ques_list.append(indexnum + 'new_how_question_from_vp' + '\t' + ques.replace(word.text, ''))
        #     print('ques_how', ques.replace(' ' + word.text, ''))
    return qw_res



def ques_word_from_pp1(pp):
    doc = nlp(pp)
    for ne in doc.ents:
        logger.debug('ne label %s \t %s ' % (ne.label_, ne.text))
        if not ne.label:
            continue
        if ne.label_ == 'PERSON':
            # add to noun_set
            noun_sing_set.add(ne.text)
            return 'who'
        if ne.label_ in ['LOC', 'GPE']:
            return 'where' # should be what since it is a NP instead of PP
        if ne.label_ in ['DATE', 'TIME']:
            return 'when'


# HEAD_WORD_FOR_HOW = ['due', 'because']
HEAD_WORD_FOR_HOW = ['using', 'by', 'through', 'with', 'via']
def ques_word_from_pp2(pp2, vp):
    if pp2[0] in HEAD_WORD_FOR_HOW:
        vp = vp[:(len(vp)-len(pp2))]
        return 'how', vp, pp2[0]
    return None, None, None




def generate_question(res,indexnum):
    text, simpleSent, np, vp, vp_tags, pp1, pp2, head_verb, head_verb_tag, head_noun_tag, reason = res['text'], res['simpleSent'], res['np'], res['vp'], res['vp_tags'], res['pp1'], res['pp2'],res['head_verb'],res['head_verb_tag'], res['head_noun_tag'], res['reason']
    doc = nlp(simpleSent)
    # print('pp2',reason,head_verb,head_noun_tag,'res',res['simpleSent'])
    # np_len = len(np)
    #
    # for word in doc[:np_len]:
    #     print 'pos', word.pos_
    #     print 'tagging',word.tag_
    #
    #     if word.ent_type:
    #
    #        print'word', word.ent_type
    #        f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/ner.txt', 'a+')
    #        f.write(str(word))
    #        f.write('\n')
    #        f.close()
    #
    # for r in textlist:




    # generate question where question word is from subject
    qw_res, subj_pron, answer_phrase = ques_word_from_subject(np, doc, res)
    if subj_pron:
        return
    for ques_word in qw_res:

        """
        res contain head_verb,  head_verb_tag, head_noun_tag, np, vp ......
        """

        main_parts = wh_sub_question_part(res)
        ques = ' '.join([ques_word, main_parts])
        ques = post_process(ques)
        ques_pool.add(indexnum+ques)
        ques_list.append(indexnum+'wh_question_from_subject' + '\t' +ques)

        l = []
        for str in simpleSent.split(" "):
        ############ low case
            if str.lower() not in ques.lower().split(" ") :
                l.append(str)
        newanswer = " ".join(l)

        # print('ques_list', ques)
        # print('newanswer', newanswer)

        answer.append(indexnum + 'subject '+newanswer)
        answer_no_index.append(newanswer)
        if len(answer_phrase) < len(ques_list):
            answer_phrase.append('answer_from_subject'+'\t'+' '.join(np))
            textlist.append(indexnum+text)
            if VERBOSE:
                #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                logger.info('NP VP(subject)\t%s' % ques)



    # wh question from pp1
    if pp1:
        ques_word = ques_word_from_pp1(pp1)
        if ques_word:
            main_parts = binary_question_part(res, PP_FLAG=False)
            ques = ' '.join([ques_word, main_parts])
            ques = post_process(ques)
            ques_pool.add(indexnum+ques)
            ques_list.append(indexnum+'wh_question_from_pp1' + '\t' +ques)
            logger.info('ques\t%s' % pp1)



            l = []
            for str in simpleSent.split(" "):
                if str.lower() not in ques.lower().split(" ") :
                    l.append(str)
            newanswer = " ".join(l)
            answer.append(indexnum + 'pp1 ' +newanswer)
            answer_no_index.append(newanswer)

            answer_phrase.append('wh_from_pp1'+'\t'+ pp1)
            logger.info('wh_from_pp1\t%s' % pp1)
            textlist.append(indexnum+text)
            # answer_phrase.append(ne)
            if VERBOSE:
                #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                logger.info('NP VP(pp1)\t%s' % ques)
    # wh question from vp
    qw_res = ques_word_from_object(np, vp, doc, reason, simpleSent, res)
    for ques_word, verbs in qw_res:
        if ques_word == False:
            ques = post_process(verbs)
            ques_pool.add(indexnum+ques)

            '''This image shows body burns how many hours prior now anasarca ?'''
            ques_list.append(indexnum+'wh_question_from_vp' + '\t' + ques)


            l = []
            for str in simpleSent.split(" "):
                if str.lower() not in ques.lower().split(" ") :
                    l.append(str)
            newanswer = " ".join(l)
            answer_phrase.append('wh from vp' + '\t' + newanswer)
            answer.append(indexnum + 'vp '+newanswer)
            answer_no_index.append(newanswer)

            textlist.append(indexnum+text)
            if VERBOSE:
                #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                logger.info('NP VP(object HARD)\t%s' % ques)
            continue
        elif ques_word == 'why':# generate why question
            new_res = copy.deepcopy(res)
            new_res['vp'] = vp[:len(vp)-len(reason)]
            main_parts = wh_obj_question_part(new_res, verbs)
        else:
            # print(res)
            # print('res', res)
            print('verbs', verbs)
            main_parts = wh_obj_question_part(res, verbs)
        # print('ques_word',ques_word)
        print('main_prats',main_parts)
        ques = ' '.join([ques_word, main_parts])
        # print(ques)
        ques = post_process(ques)
        ques_pool.add(indexnum+ques)

        ques_list.append(indexnum+ 'rest_wh_question_from_vp' + '\t' +ques)
        np_len = len(np)
        # print('wh_question_from_vp', ques)
        if len(answer_phrase) < len(ques_list):
            # print(('docr',doc[np_len+1:]))
            answer_phrase.append('answer_rest_wh_question_from_vp' + '\t' + doc[np_len+1:].text)
            # print('wh from vp', doc[np_len+1:].text)


        l = []
        for str in simpleSent.split(" "):
            if str.lower() not in ques.lower().split(" ") :
                l.append(str)
        newanswer = " ".join(l)
        answer.append(indexnum + 'vp2 '+newanswer)
        answer_no_index.append(newanswer)

        ##### words after what
        np_len = len(np)
        vp_len = len(vp)

        textlist.append(indexnum+text)
        if VERBOSE:
            #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
            logger.info('NP VP(object)\t%s' % ques)
    # how question from pp2
    if pp2:
        ques_word, verbs, ne = ques_word_from_pp2(pp2, vp)
        if ques_word:
            new_res = copy.deepcopy(res)
            new_res['vp'] = verbs
            main_parts = binary_question_part(new_res)
            ques = ' '.join([ques_word, main_parts])
            ques = post_process(ques)
            ques_pool.add(indexnum+ques)
            ques_list.append(indexnum+'how_question_from_pp2' + '\t' +ques)

            l = []
            for str in simpleSent.split(" "):
                if str.lower() not in ques.lower().split(" ") :
                    l.append(str)
            newanswer = " ".join(l)
            answer.append(indexnum + 'answer_pp2 ' + newanswer)
            answer_no_index.append(newanswer)
            # print('how_question_from_pp2', ques, ' '.join(newanswer.split(" ")[1:]))
            answer_phrase.append('answer_how_pp2'+'\t'+ ' '.join(newanswer.split(" ")[1:]))
            # print('answer pp2', ' '.join(pp2))
            textlist.append(indexnum+text)
            if VERBOSE:
                #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                logger.info('NP VP(pp2)\t%s' % ques)


    # yes-no question
    if not subj_pron:
        # make yes question
        main_parts = binary_question_part(res)

        logger.info('Yes main parts: %s', main_parts)

        ques = post_process(main_parts)
        ques_pool.add(indexnum+ques)
        ques_list.append(indexnum+'yes_question' + '\t' +ques)
        # print('ques',ques)
        answer.append(indexnum+'yes')
        answer_no_index.append('yes')
        answer_phrase.append('answer_yes_question' + '\t' + 'yes')
        # print('answer_phrase', answer_phrase)
        textlist.append(indexnum+text)
        if VERBOSE:
            #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
            logger.info('NP VP(yes)\t%s' % ques)
        # make no question
        np_doc = nlp(' '.join(np))
        is_plural = False
        for n in np_doc:
            if n.tag_.endswith('S'):
                is_plural = True
        if is_plural and noun_plu_set:
            new_res = copy.deepcopy(res)
            new_np = random.sample(noun_plu_set, 1)
            if ' '.join(res['np']).lower() != new_np[0]:
                new_res['np'] = new_np
                logger.info('np1 %s np2 %s' % (res['np'], new_res['np']))
                main_parts = binary_question_part(new_res)
                ques = post_process(main_parts)
                ques_pool.add(indexnum+ques)
                ques_list.append(indexnum+'no_question' + '\t' +ques)
                answer.append(indexnum+'no')
                answer_no_index.append('no')
                answer_phrase.append('answer_no_question' +'\t' + 'no')
                textlist.append(indexnum+text)
                if VERBOSE:
                    #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                    logger.info('NP VP(no)\t%s' % ques)
        elif not is_plural and noun_sing_set:
            new_res = copy.deepcopy(res)
            new_np = random.sample(noun_sing_set, 1)
            if ' '.join(res['np']).lower() != new_np[0]:
                new_res['np'] = new_np
                # logger.info('np1 %s np2 %s' % (' '.join(res['np']).lower(), new_np[0]))
                main_parts = binary_question_part(new_res)
                ques = post_process(main_parts)
                ques_pool.add(indexnum+ques)
                ques_list.append(indexnum+'no_question_not_plural' + '\t' +ques)
                logger.info('no_question_not_plural %s' % ques)
                answer.append(indexnum+'no')
                answer_no_index.append('no')
                answer_phrase.append('answer_no_question_not_plural'+'\t'+'no')

                # print('no_question_not_plural',ques)
                # print('answer_no_question_not_plural', 'no')
                textlist.append(indexnum+text)
                if VERBOSE:
                    #logger.info('\t%s \t \t%s \t \t%s' % (simpleSent, np, vp))
                    logger.info('NP VP(no)\t%s' % ques)


def binary_question_part(res, PP_FLAG=True):
    simpleSent, np, vp, vp_tags, pp1, head_verb, head_verb_tag, head_noun_tag = res['simpleSent'], res['np'], res['vp'], res['vp_tags'], res['pp1'], res['head_verb'],res['head_verb_tag'], res['head_noun_tag']
    has_aux = has_auxiliary(head_verb, head_verb_tag, vp_tags)
    if has_aux:
        if PP_FLAG:
            logger.info('np:%s vp:%s pp1:%s' % (np, vp, pp1))
            return ' '.join([head_verb, ' '.join(np), pp1, ' '.join(vp[1:]) ])
        else:
            return ' '.join([head_verb, ' '.join(np), ' '.join(vp[1:])])
    else:
        do, verb = decompose_verb(head_verb, head_verb_tag)
        if PP_FLAG:

            logger.info('d0:%s np:%s vp:%s pp1:%s' % (do, np ,vp , pp1))
            return ' '.join([do, ' '.join(np),  verb, ' '.join(vp[1:]), pp1])
        else:
            logger.info('np:%s vp:%s pp1:%s' % (np, vp, pp1))
            return ' '.join([do, ' '.join(np), verb, ' '.join(vp[1:])])


def wh_sub_question_part(res):
    simpleSent, np, vp, vp_tags, pp1, head_verb, head_verb_tag, head_noun_tag = res['simpleSent'], res['np'], res['vp'], res['vp_tags'], res['pp1'], res['head_verb'],res['head_verb_tag'], res['head_noun_tag']
    has_aux = has_auxiliary(head_verb, head_verb_tag, vp_tags)
    if has_aux:
        # print('whatjoin', ' '.join([head_verb, ' '.join(vp[1:]), pp1]))
        # print('whatjoinnp', np)
        return ' '.join([head_verb, ' '.join(vp[1:]), pp1])
    else:
        #do, vp[0] = decompose_verb(head_verb, head_verb_tag)
        # print('whatjoin2', ' '.join([' '.join(vp), pp1]))
        return ' '.join([' '.join(vp), pp1])

def wh_obj_question_part(res, verbs):
    simpleSent, np, vp, vp_tags, pp1, head_verb, head_verb_tag, head_noun_tag = res['simpleSent'], res['np'], res['vp'], res['vp_tags'], res['pp1'], res['head_verb'],res['head_verb_tag'], res['head_noun_tag']
    has_aux = has_auxiliary(head_verb, head_verb_tag, vp_tags)
    if has_aux:
        return ' '.join([head_verb, ' '.join(np), verbs, pp1])
    else:
        do, head_verb = decompose_verb(head_verb, head_verb_tag)
        # return ' '.join([do, ' '.join(np), head_verb, verbs, pp1])
        return ' '.join([do, ' '.join(np), head_verb])
        ###### doe this image show

def why_obj_question_part(res, verbs):
    simpleSent, np, vp, vp_tags, pp1, head_verb, head_verb_tag, head_noun_tag = res['simpleSent'], res['np'], res['vp'], \
                                                                                res['vp_tags'], res['pp1'], res[
                                                                                    'head_verb'], res['head_verb_tag'], \
                                                                                res['head_noun_tag']
    do, head_verb = decompose_verb(head_verb, head_verb_tag)
    return ' '.join([do, ' '.join(np), head_verb, verbs, pp1])
    # return ' '.join([do, ' '.join(np), head_verb])

def post_process(ques):
    ques = re.sub(' +',' ',ques) + '?'
    ques = ques[0].capitalize() + ques[1:]
    ques = re.sub('-LRB- ', '', ques, flags=re.IGNORECASE)
    ques = re.sub(' *-RRB-', '', ques, flags=re.IGNORECASE)
    ques = re.sub('`` ', '"', ques)
    ques = re.sub(" ''", '"', ques)
    return ques

def post_process_QA(str):
    str = re.sub('-LRB- ', '', str, flags=re.IGNORECASE)
    str = re.sub(' *-RRB-', '', str, flags=re.IGNORECASE)
    return str

def post_process_ques(ques):
    ques.replace(' ?', '?')
    ques.replace(' ,', ',')
    return ques

def post_process_answer(ans):
    ans = re.sub('`` ', '"', ans)
    ans = re.sub(" ''", '"', ans)
    ans = re.sub(' ,', ',', ans)
    return ans


# check if plural
def is_plural(tag):
    if not tag:
        return False
    return tag.endswith('S')

# check if possessive pronoun
def is_possessive(word):
    return word.text in ['his', 'her', 'their']

ADV_STOP_LIST = ['almost', 'also', 'further', 'generally', 'greatly','however', 'just', 'later', 'longer', 'often', 'only', 'typically']
ADV_LIST = ['by', 'via', 'through']
def is_how(word):
    return word in ADV_LIST

# check if contains auxiliary
def has_auxiliary(head_verb, head_verb_tag, vp_tags):
    if head_verb_tag=='MD' or lemma(head_verb) in ['be','do']:
        return True
    if lemma(head_verb) in ['have'] and len(vp_tags) > 2 and vp_tags[2].startswith('V'):
        return True
    return False


def decompose_verb(verb, verb_tag):
    logger.debug('verb \t %s verb_tag \t %s' % (verb, verb_tag))
    tense = verb_tense_dict[verb_tag]
    if conjugate('do', tense) == 'did':
        return_do = 'is'
        return return_do, verb
    else:
        return_do = conjugate('do', tense)
    # elif conjugate('do', tense) == does:
        print('verb',verb, verb_tag)
        print(conjugate('do', tense))
        print('lemma',lemma(verb))
        return return_do, lemma(verb)


def get_words_and_tags(tree, TAG=False):
    """
    Return (words, tags) if TAG is True else False
    """
    if TAG:
        return (re.findall(r'(?<= )?[^( )]+(?=\))', tree), re.findall(r'(?<=\()\w+(?= )',tree))
    else:
        return re.findall(r'(?<= )?[^( )]+(?=\))', tree)




def ask(farticle, nquestions):
    article = Article(farticle)
    sentences = article.getRawLines()
    if VERBOSE:
        logger.debug('sentences\t%slen%d' % (sentences[0],len(sentences)))
    # ignore super long sentences (more than 50 words)
    sentences = [s.strip() for s in sentences if s.count(' ') < 50]
    sentences = ['793    On the T2-weighted images, there are curvilinear serpentine hypointensities, suggestive of vessels, hemosiderin deposition, or calcification.']
    for sent in sentences:
        sent = sent.encode('ascii', 'ignore').decode('ascii')
        sent = re.sub(u'\(.*\) ','', sent)
        # simplify sentence
        for (TYPE, pattern) in patterns:
            # print 'sent \t%s pattern \t%s TYPE \t%s' % (sent, pattern, TYPE)
            getSimpleSentence(sent, pattern, TYPE)

    # rankedQues = question_evaluator.get_score(ques_pool, nquestions)



    dir = '/Users/jkooy/research/self_dataset/crawled_modified_captions/test_small/new_test/part1/'


    # for q in rankedQues:
    #
    #     f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/question.txt', 'a+')
    #     f.write(q)
    #     f.write('\n')
    #     f.close()

    # print 'total ques', len(ques_pool)

    fq = open(dir + 'question_list_new.txt', 'a+')
    fq.write('Images' + '\t' + 'Qtypes' + '\t' +'Questions' + '\t' + 'Atypes' + '\t' +'Answers')
    fq.write('\n')
    fq.close()

    for q, a in zip(ques_list,answer_phrase):
        # f = open('out_question_list.txt', 'a+')
        # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/question_list_new.txt', 'a+')
        # print('a',a)
        # print('q', q)
        a_split = []
        for str in a.split(" "):

            if str.lower() not in q.lower().split(" ") :
                a_split.append(str)
        a = " ".join(a_split)

        a = re.sub('-LRB- ', '', a, flags=re.IGNORECASE)
        a = re.sub(' *-RRB-', '', a, flags=re.IGNORECASE)
        if a != 'answer_from_subject'+'\t'+'this image':
            fq = open(dir + 'question_list_new.txt', 'a+')
            fq.write(post_process_ques(q) + '\t' + post_process_answer(a))
            fq.write('\n')
            fq.close()


    # for q in ques_list:
    #     # f = open('out_question_list.txt', 'a+')
    #     # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/question_list_new.txt', 'a+')
    #     f = open(dir +'question_list_new.txt', 'a+')
    #     f.write(q)
    #     f.write('\n')
    #     f.close()


    for p in answer:
        p = re.sub('-LRB- ', '', p, flags=re.IGNORECASE)
        p = re.sub(' *-RRB-', '', p, flags=re.IGNORECASE)
        p = re.sub('shows', '', p, flags=re.IGNORECASE)
        p = re.sub('illustrates', '', p, flags=re.IGNORECASE)
        # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/answer_list_new.txt', 'a+')
        f = open(dir +'answer_list_new.txt', 'a+')
        f.write(p)
        f.write('\n')
        f.close()

    answer_no_index_postprocessing = []
    for m in answer_no_index:
        m = re.sub('-LRB- ', '', m, flags=re.IGNORECASE)
        m = re.sub(' *-RRB-', '', m, flags=re.IGNORECASE)
        m = re.sub('shows', '', m, flags=re.IGNORECASE)
        m = re.sub('illustrates', '', m, flags=re.IGNORECASE)
        # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/answer_list_new.txt', 'a+')
        # f = open('/Users/jkooy/research/self_dataset/crawled_modified_captions/new_version/answer_frequency_new.txt', 'a+')
        answer_no_index_postprocessing.append(m)
        # f.write(m)
        # f.write('\n')
        # f.close()


    # answer_frequency = Counter(answer_no_index_postprocessing)
    # print(answer_frequency)
    # answer_frequency_sort = sorted(answer_frequency.items(), key=lambda x: x[1], reverse=True)
    # print(answer_frequency_sort)
    # for m in answer_frequency_sort:
    #     f = open(dir +'answer_frequency.txt', 'a+')
    #     f.write(str(m[0])+'\t'+str(m[1]))
    #     f.write('\n')
    #     f.close()

    # for r in textlist:
    #     # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/text_list_new.txt', 'a+')
    #     f = open('/Users/jkooy/research/self_dataset/crawled_modified_captions/test_small/text_list_new.txt', 'a+')
    #     f.write(r)
    #     f.write('\n')
    #     f.close()

    # for m in answer_phrase:
    #     # f = open('/Users/jkooy/research/caption_image_extract/columnexcel/pathology_textbook_single/output/answer_new.txt', 'a+')
    #     f = open(dir +'answer_new.txt', 'a+')
    #     f.write(str(m))
    #     f.write('\n')
    #     f.close()



# farticle = '/Users/jkooy/research/self_dataset/crawled_modified_captions/test_small/caption_crawled_150.txt'
# farticle = '/Users/jkooy/research/self_dataset/crawled_modified_captions/test_small/caption_crawled_1.txt'
farticle = '/Users/jkooy/research/self_dataset/crawled_modified_captions/caption_crawled_cc3.txt'
# farticle = '/Users/jkooy/research/self_dataset/crawled_modified_captions/caption_crawled_original.txt'
# farticle = '/Users/jkooy/research/self_dataset/crawled_modified_captions/new_version/caption_crawled_cc3.txt'
# farticle = '/Users/jkooy/research/self_dataset/rad_dataset/open_i/MRI/caption_crawled_1.txt'
# farticle = '/Users/jkooy/research/self_dataset/rad_dataset/open_i/ct/caption_all_ct.txt'
# farticle = '/Users/jkooy/research/self_dataset/rad_dataset/open_i/x-ray/caption_all_xray.txt'
nquestions = 100000
print 'Usage: ./ask.py article.txt nqeustions'
ask(farticle, nquestions)

# farticle = sys.argv[1]
# nquestions = int(sys.argv[2])
# if len(sys.argv)<3:
#     print 'Usage: ./ask.py article.txt nqeustions'
# ask(farticle, nquestions)



