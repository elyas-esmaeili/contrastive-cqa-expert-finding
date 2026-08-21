#!/home/zeyu/anaconda3/bin/python3.6

"""
    Preprocessing

    Author:
        Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@ucla.edu>

    Description:
        Take in the Very raw data and produce a ready-to-use version
"""
import sentence_transformers
from sentence_similarity import top_k_simillar
from tqdm.auto import tqdm


import sys, os
import re
import logging
import numpy as np
from lxml import etree
from bs4 import BeautifulSoup

import nltk
nltk.data.path.append("/workspace/nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string, random
from collections import Counter 

try:
    import ujson as json
except:
    import json

# Users participated in Asking and Answering
part_user = set()
a = 0

# count how many questions an users asked
# count how many questions an answerer responded
count_Q, count_A = {}, {}

qa_map = {}
test_candidates = set()

def clean_html(x):
    return BeautifulSoup(x, 'lxml').get_text()


def clean_str(string):
    """Clean up the string

    Cleaning strings of content or title
    Original taken from [https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py]

    Args:
        string - the string to clean

    Return:
        _ - the cleaned string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str2(s):
    """Clean up the string

    * New version, removing all punctuations

    Cleaning strings of content or title
    Original taken from [https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py]

    Args:
        string - the string to clean

    Return:
        _ - the cleaned string
    """
    ss = s
    translator = str.maketrans("", "", string.punctuation)
    ss = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", ss)
    ss = re.sub(r"\'s", "s", ss)
    ss = re.sub(r"\'ve", "ve", ss)
    ss = re.sub(r"n\'t", "nt", ss)
    ss = re.sub(r"\'re", "re", ss)
    ss = re.sub(r"\'d", "d", ss)
    ss = re.sub(r"\'ll", "ll", ss)
    ss = re.sub(r"\s{2,}", " ", ss)
    ss = ss.translate(translator)
    return ss.strip().lower()


def remove_stopwords(string, stopword_set):
    """Removing Stopwords

    Args:
        string - the input string to remove stopwords
        stopword_set - the set of stopwords

    Return:
        _ - the string that has all the stopwords removed
    """
    word_tokens = word_tokenize(string)
    filtered_string = [word for word in word_tokens
                       if word not in stopword_set]
    return " ".join(filtered_string)


def split_post(raw_dir, data_dir):
    """ Split the post

    Split post to question and answer,
    keep all information, output to file

    Args:
        raw_dir - raw data directory
        data_dir - parsed data directory
    """
    if os.path.exists(data_dir + "Posts_Q.json") \
        and os.path.exists(data_dir + "Posts_A.json"):
        print("\t\tPosts_Q.json, Posts_A.json already exists."
              "Skipping the split_post.")
        return

    with open(data_dir + "Posts_Q.json", "w") as fout_q, \
            open(data_dir + "Posts_A.json", "w") as fout_a:
        parser = etree.iterparse(raw_dir + 'Posts.xml',
                                 events=('end',), tag='row')
        for event, elem in parser:
            attr = dict(elem.attrib)
            attr['Body'] = clean_html(attr['Body'])

            # Output to separate files
            if attr['PostTypeId'] == '1':
                fout_q.write(json.dumps(attr) + "\n")
            elif attr['PostTypeId'] == '2':
                fout_a.write(json.dumps(attr) + "\n")
    return


def process_QA(data_dir):
    """Process QA

    Extract attributes used in this project
    Get rid of the text information,
    only record the question-user - answer-user relation

    Args:
        data_dir - the dir where primitive data is stored
    """
    POST_Q = "Posts_Q.json"
    POST_A = "Posts_A.json"
    OUTPUT = "Record_All.json"
    RAW_STATS = "question.stats.raw"

    # Get logger to log exceptions
    logger = logging.getLogger(__name__)

    no_acc_question = 0

    raw_question_stats = []

    if not os.path.exists(data_dir + POST_Q):
        raise IOError("file {} does NOT exist".format(data_dir + POST_Q))

    if not os.path.exists(data_dir + POST_A):
        raise IOError("file {} does NOT exist".format(data_dir + POST_A))

    # Process question information
    with open(data_dir + POST_Q, 'r') as fin_q:
        for line in fin_q:
            data = json.loads(line)
            try:
                qid, rid = data.get('Id', None), data.get('OwnerUserId', None)
                # If such 
                if qid and rid:
                    acc_id = data.get('AcceptedAnswerId', None)
                    answer_count = int(data.get('AnswerCount', -1))
                    tags_str = data.get('Tags', "")
                    # tags = re.findall(r"<([^>]+)>", tags_str)
                    tags = tags_str.split("|")[1:-1]
                    if acc_id:
                        qa_map[qid] = {
                            'QuestionId': qid,
                            'QuestionOwnerId': rid,
                            'AcceptedAnswerId': acc_id,
                            'AcceptedAnswererId': None,
                            'AnswererIdList': [],
                            'AnswererAnswerTuples': [],
                            'Tags': tags
                        }
                        count_Q[rid] = count_Q.get(rid, 0) + 1
                    else:
                        no_acc_question += 1

                    if answer_count >= 0:
                        raw_question_stats.append(answer_count)
            except:
                logger.error("Error at process_QA 1: " + str(data))
                continue
    print("\t\t{} questions do not have accepted answer!"
          .format(no_acc_question))
    
    # Count raw question statistics
    raw_question_stats_cntr = Counter(raw_question_stats)
    with open(data_dir + RAW_STATS, "w") as fout:
        for x in sorted(list(raw_question_stats_cntr.keys())):
            print("{}\t{}".format(x, raw_question_stats_cntr[x]), file=fout)
        print("Total\t{}".format(sum(raw_question_stats)), file=fout)

    # Process answer information
    with open(data_dir + POST_A, 'r') as fin_a:
        for line in fin_a:
            data = json.loads(line)
            try:
                answer_id = data.get('Id', None)
                aid = data.get('OwnerUserId', None)
                qid = data.get('ParentId', None)
                entry = qa_map.get(qid, None)

                # If AcceptedAnswererId doesn't exist, delete question from qa_map
                if answer_id and qid and entry and (not aid):
                    if answer_id == entry['AcceptedAnswerId']:
                        for user in entry["AnswererIdList"]:
                            count_A[user] -= 1
                        del qa_map[qid]
                        continue

                if answer_id and aid and qid and entry:
                    entry['AnswererAnswerTuples'].append((aid, answer_id))
                    entry['AnswererIdList'].append(aid)
                    count_A[aid] = count_A.get(aid, 0) + 1

                    # Check if we happen to hit the accepted answer
                    if answer_id == entry['AcceptedAnswerId']:
                        entry['AcceptedAnswererId'] = aid
                else:
                    logger.error(
                        "Answer {} belongs to unknown Question {} at Process QA"
                        .format(answer_id, qid))
            except IndexError as e:
                logger.error(e)
                logger.info("Error at process_QA 2: " + str(data))
                continue

    # Fill in the blanks of `AcceptedAnswererId`
    # for qid in qa_map.keys():
    #    acc_id = qa_map[qid]['AcceptedAnswerId']
    #    for aid, answer_id in qa_map[qid]['AnswererAnswerTuples']:
    #        if answer_id == acc_id:
    #            qa_map[qid]['AcceptedAnswererId'] = aid
    #            break

    print("\t\tWriting the Record for ALL to disk.")
    with open(data_dir + OUTPUT, 'w') as fout:
        for q in qa_map.keys():
            fout.write(json.dumps(qa_map[q]) + "\n")


def question_stats(data_dir):
    """Find the question statistics for `Introduction`

    Args:
        data_dir -
    Return
    """
    OUTPUT = "question.stats"
    count = []
    for qid in qa_map.keys():
        ans_count = len(qa_map[qid]['AnswererIdList'])
        count.append(ans_count)
        if ans_count == 0:
            pass
            # print("0 answer id list", qid)
    question_stats_cntr = Counter(count)

    with open(data_dir + OUTPUT, "w") as fout:
        for x in sorted(list(question_stats_cntr.keys())):
            print("{}\t{}".format(x, question_stats_cntr[x]), file=fout)
        print("Total\t{}".format(sum(count), file=fout), file=fout)
    return


def answerer_stats(data_dir):
    OUTPUT = "answerer.stats"
    with open(data_dir + OUTPUT, "w") as fout:
        for (aid, n_answered_questions) in count_A.items():
            print(f"{aid} {n_answered_questions}", file=fout)


def generate_pmef_dataset(data_dir, parsed_dir):
    INPUT_TITLE = "Q_title.txt"
    INPUT_BODY = "Q_content.txt"
    INPUT = "Record_All.json"
    
    if not os.path.exists(f"{data_dir}/PMEF"):
        os.makedirs(f"{data_dir}/PMEF")
        os.makedirs(f"{data_dir}/PMEF/train")
        os.makedirs(f"{data_dir}/PMEF/dev")
        
    raw_question_tags = dict()
    with open(data_dir + INPUT, "r") as fin:
        for line in fin:
            data = json.loads(line)
            qid = data['QuestionId']
            tags = data['Tags']
            raw_question_tags[qid] = tags

    import itertools
    all_tags = list(itertools.chain(*list(raw_question_tags.values())))
    all_tags = list(set(all_tags))
    print(f"Number of tags: {len(all_tags) + 1}")
    nn_tags = len(all_tags)
    tags_to_index = {k: v for v, k in enumerate(all_tags)}
    tags_to_index["<PAD>"] = max(list(tags_to_index.values())) + 1


    raw_question_titles = dict()
    with open(parsed_dir + INPUT_TITLE, "r") as fin:
        for line in fin:
            try:
                qid, content = line.split(maxsplit=1)
            except:
                continue
            raw_question_titles[qid] = content

    raw_question_bodies = dict()
    with open(parsed_dir + INPUT_BODY, "r") as fin:
        for line in fin:
            qid, content = line.split(maxsplit=1)
            raw_question_bodies[qid] = content

    question_titles = {}
    question_bodies = {}
    question_tags = {}
    qid_to_index = {}  # Mapping from qid to index in the matrix
    for index, (qid, title) in enumerate(raw_question_titles.items()):
        qid_to_index[qid] = index
        title_words = title.split()
        question_titles[index] = " ".join(title_words[:min(len(title_words), 15)] + ["<PAD>"] * (15 - len(title_words)))

        body = raw_question_bodies[qid]
        body_words = body.split()
        question_bodies[index] = " ".join(body_words[:min(len(body_words), 60)] + ["<PAD>"] * (60 - len(body_words)))

        tags = [tags_to_index[tag] for tag in raw_question_tags[qid]]
        tags = tags[:min(3, len(tags))] + [tags_to_index["<PAD>"]] * (3 - len(tags))
        question_tags[index] = tags
    
    max_index = max(list(qid_to_index.values()))
    question_titles[max_index + 1] = " ".join(["<PAD>"] * 15)
    question_bodies[max_index + 1] = " ".join(["<PAD>"] * 60)
    question_tags[max_index + 1] = [tags_to_index["<PAD>"]] * 3

    np.save(f"{data_dir}/PMEF/q_title.npy", question_titles)
    np.save(f"{data_dir}/PMEF/q_body.npy", question_bodies)
    np.save(f"{data_dir}/PMEF/q_tag.npy", question_tags)

    return qid_to_index, max_index + 1, question_tags, tags_to_index

def build_test_set(data_dir, parsed_dir, threshold, test_sample_size,
                   test_proportion):
    """
    Building test datase,
    test_proportiont
    Args:
        parse_dir - the directory to save parsed set.
        threshold - the selection threshold

    Return:
    """
    import pandas as pd
    votes_df = pd.read_csv(data_dir+"votes.csv", index_col=0)
    INPUT_C = "Q_title.txt"
    INPUT_ENC = "Q_encoding.npy"

    if not os.path.exists(parsed_dir + INPUT_C):
        IOError("Can not locate {}".format(parsed_dir + INPUT_C))


    qid_to_index, pad_id, question_tags, tag_index = generate_pmef_dataset(data_dir, parsed_dir)


    questions = dict()
    with open(parsed_dir + INPUT_C, "r") as fin:
        for line in fin:
            try:
                qid, content = line.split(maxsplit=1)
            except:
                continue
            questions[qid] = content

    questions_encoding = np.load(parsed_dir + INPUT_ENC, allow_pickle=True)
    questions_encoding = questions_encoding.item()

    # Normalize embeddings
    # normalized_embeddings = []
    # question_id_to_index = {}  # Mapping from qid to index in the matrix
    # for index, (qid, emb) in enumerate(questions_encoding.items()):
    #     normalized_embeddings.append(emb / np.linalg.norm(emb))
    #     question_id_to_index[qid] = index

    # # Create a similarity matrix
    # embedding_matrix = np.stack(normalized_embeddings)
    # similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

    # def count_similar_questions(user_questions, asked_qid):
    #     asked_q_index = question_id_to_index[asked_qid]
    #     count = 0
    #     for qid in user_questions:
    #         if qid in question_id_to_index:
    #             q_index = question_id_to_index[qid]
    #             if similarity_matrix[asked_q_index, q_index] >= 0.1:
    #                 count += 1
    #     return count


    with open(parsed_dir + "answerer_stats_2.json") as fin:
        aid_qid_list = json.load(fin)

    TEST = "test.txt"
    OUTPUT_TRAIN = "Record_Train.json"

    accept_no_answerer = 0

    ordered_count_A = sorted(
        count_A.items(), key=lambda x:x[1], reverse=True)
    # ordered_aid = [x[0] for x in ordered_count_A if x[1] >= 5]
    ordered_aid = [x[0] for x in ordered_count_A]

    ordered_aid = ordered_aid[: int(len(ordered_aid) * 0.1)]

    question_count = len(qa_map)

    for qid in qa_map.keys():
        accaid = qa_map[qid]['AcceptedAnswererId']
        rid = qa_map[qid]['QuestionOwnerId']
        if not accaid:
            accept_no_answerer += 1
            continue
        if count_Q[rid] >= threshold and count_A[accaid] >= threshold:
            test_candidates.add(qid)

    print("\t\tSample table size {}. Using {} instances for test."
          .format(len(test_candidates), int(question_count * test_proportion)))

    # test = np.random.choice(list(test_candidates),
    #                         size=int(question_count * test_proportion),
    #                         replace=False)
    n_test = int(question_count * test_proportion)
    list_test_candidates = list(test_candidates)
    list_test_candidates.sort(key=int, reverse=True)
    test = list_test_candidates[:n_test]

    print("\t\tAccepted answer without Answerer {}".format(accept_no_answerer))
  
    aid_to_index = {k: v for v, k in enumerate(list(aid_qid_list.keys()))}
    a_qid = {}
    # pad_id = #
    for uid in aid_qid_list:
        a_qids = aid_qid_list[uid]
        a_qids_index = [qid_to_index[qid] for qid in a_qids if qid not in test]
        a_qid[aid_to_index[uid]] = a_qids_index[:min(30, len(a_qids_index))] + [pad_id] * (30 - len(a_qids_index))

    test_dict = {"qid": [], "qtags": [], "label": [], "question": [], "answerer": [], "answerer_tags": []}
    pmef_dataset_dict_train = {"qid": [], "qid_list": [], "aid_list": [], "label_list": []}

    train_dict = {"qid": [], "qtags": [], "label": [], "question": [], "answerer": [], "answerer_tags": []}
    pmef_dataset_dict_test = {"qid": [], "qid_list": [], "aid_list": [], "label_list": []}

    
    def add_samples_to_dataset(question_id, dataset_dict, pmef_dataset_dict, train=True):
        global a
        asked_question_id = question_id # Replace with the actual qid
        # user_similarity_counts = dict()
        # for user in ordered_aid:
        #     # print(user, type(user))
        #     user_similarity_counts[user] = count_similar_questions(aid_qid_list[str(user)], asked_question_id)

        # inverse_freq_scores = {user: 1.0 / count if count > 0 else float('inf')
        #             for user, count in user_similarity_counts.items()}
        # sorted_users = sorted(inverse_freq_scores, key=inverse_freq_scores.get, reverse=True)
        # N = 20  # Number of negative samples
        # negative_samples = sorted_users[:N]
        # if accaid not in negative_samples:
        #     negative_samples[random.randint(0, 19)] = accaid

        # train_s.extend(negative_samples)
        # REgression
        negative_samples = random.sample(ordered_aid, 20)
        if accaid not in negative_samples:
            negative_samples[random.randint(0, 19)] = accaid
        pmef_samples = negative_samples.copy()

        if train:
            negative_samples = qa_map[asked_question_id]["AnswererIdList"]
            answerer_answer_id = dict(qa_map[asked_question_id]["AnswererAnswerTuples"])
            # REgression
        
        samples = negative_samples
        # Random


        # qid = question_id
        # rid = qa_map[qid]['QuestionOwnerId']
        # accaid = qa_map[qid]['AcceptedAnswererId']
        # aid_list = qa_map[qid]['AnswererIdList']
        # if len(aid_list) <= test_sample_size:
        #     neg_sample_size = test_sample_size - len(aid_list)
        #     neg_samples = random.sample(ordered_aid, neg_sample_size)
        #     samples = neg_samples + aid_list
        # else:
        #     samples = random.sample(aid_list, test_sample_size)
        #     if accaid not in samples:
        #         samples.pop()
        #         samples.append(accaid)

        for sample in samples:
            dataset_dict["qid"].append(int(qid))
            dataset_dict["qtags"].append(question_tags[qid_to_index[qid]])

            dataset_dict["question"].append(questions[qid].strip())
            list_answered_question = list(set(aid_qid_list[str(sample)]) - set(test))
            how_many = min(30, len(list_answered_question))
            # top_k = random.sample(list_answered_question, how_many)
            
            	
            top_k = top_k_simillar(qid, list_answered_question, questions_encoding, 30)
            # if len(top_k) == 0:
            #     print(sample, len(aid_qid_list[str(sample)]),len(list_answered_question))
            dataset_dict["answerer"].append([questions[str(idx)].strip() for idx in top_k])
            dataset_dict["answerer_tags"].append([question_tags[qid_to_index[idx]] for idx in top_k])
            # print(len(top_k))
            for i in range(len(top_k), 30):
                # print("yes")
                dataset_dict["answerer_tags"][-1].append([tag_index["<PAD>"], tag_index["<PAD>"], tag_index["<PAD>"]])

            if not train:
                dataset_dict["label"].append(int(sample == accaid))
            else:
                try:
                    dataset_dict["label"].append(votes_df.loc[int(answerer_answer_id[sample]), "votes"])
                except Exception as e:
                    a += 1
                    dataset_dict["label"].append(0)
                # print(e)
                # print("sample", sample, type(sample))
                # print("answerer_answer_id: ", answerer_answer_id[sample], type(answerer_answer_id[sample]))
                # print(answerer_answer_id[sample])

        for sample in pmef_samples:
            pmef_dataset_dict["qid_list"].append(qid_to_index[qid])
            pmef_dataset_dict["aid_list"].append(aid_to_index[sample])
            pmef_dataset_dict["label_list"].append(int(sample==accaid))

    progress_bar = tqdm(range(len(test)))
    print("\t\tWriting the sampled test set to disk")
    with open(parsed_dir + TEST, "w") as fout:
        for qid in test:
            rid = qa_map[qid]['QuestionOwnerId']
            accaid = qa_map[qid]['AcceptedAnswererId']
            aid_list = qa_map[qid]['AnswererIdList']
            pmef_dataset_dict_test["qid"].append(qid_to_index[qid])
            add_samples_to_dataset(qid, test_dict, pmef_dataset_dict_test, train=False)

            progress_bar.update(1)

    # if qid is a test instance or qid doesn't have an answer
    qid_list = list(qa_map.keys())
    for qid in qid_list:
        if qid in test\
            or not len(qa_map[qid]['AnswererIdList'])\
            or not qa_map[qid]['AcceptedAnswererId']:
            del qa_map[qid]

    progress_bar = tqdm(range(len(list(qa_map.keys()))), position=0, leave=True)
    with open(parsed_dir + TEST, "w") as fout:
        for qid in qa_map.keys():
            rid = qa_map[qid]['QuestionOwnerId']
            accaid = qa_map[qid]['AcceptedAnswererId']
            aid_list = qa_map[qid]['AnswererIdList']

            pmef_dataset_dict_train["qid"].append(qid_to_index[qid])
            add_samples_to_dataset(qid, train_dict, pmef_dataset_dict_train, train=True)

            progress_bar.update(1)

    for file_name in pmef_dataset_dict_train:
        np.save(f"{data_dir}/PMEF/train/{file_name}.npy", pmef_dataset_dict_train[file_name])

    for file_name in pmef_dataset_dict_test:
        np.save(f"{data_dir}/PMEF/dev/{file_name}.npy", pmef_dataset_dict_test[file_name])

    np.save(f"{data_dir}/PMEF/aid.npy",list(aid_to_index.values()))
    np.save(f"{data_dir}/PMEF/a_qid.npy", a_qid)

    with open(data_dir + "train_dict.json", 'w') as fout: 
        fout.write(json.dumps(train_dict))

    
    with open(data_dir + "test_dict.json", 'w') as fout: 
        fout.write(json.dumps(test_dict))
    print("\n",a)


    # # Write QA pair to file
    # print("\t\tWriting the Record for training to disk")
    # with open(data_dir + OUTPUT_TRAIN, 'w') as fout:
    #     for q in qa_map.keys():
    #         fout.write(json.dumps(qa_map[q]) + "\n")
    return


def extract_question_user(data_dir, parsed_dir):
    """Extract Question User pairs and output to file.
    Extract "Q" and "R". Format:
        <Qid> <Rid>
    E.g.
        101 40
        145 351

    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    # INPUT = "Record_Train.json"
    INPUT = "Record_All.json"
    OUTPUT = "Q_R.txt"

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT find {}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin:
        with open(parsed_dir + OUTPUT, "w") as fout:
            for line in fin:
                data = json.loads(line)
                qid = data['QuestionId']
                rid = data['QuestionOwnerId']
                part_user.add(int(rid))  # Adding participated questioners
                print("{} {}".format(str(qid), str(rid)), file=fout)


def extract_question_tags(data_dir, parsed_dir):
    """Extract Question User pairs and output to file.
    Extract "Q" and "R". Format:
        <Qid>, <Tags>
    E.g.
        10 java jar jdk
        101 python keras

    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    # INPUT = "Record_Train.json"
    INPUT = "Record_All.json"
    OUTPUT = "Q_Tags.txt"

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT find {}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin:
        with open(parsed_dir + OUTPUT, "w") as fout:
            for line in fin:
                data = json.loads(line)
                qid = data['QuestionId']
                tags = data['Tags']
                print("{}, {}".format(str(qid), " ".join(tags)), file=fout)


def extract_question_answer_user(data_dir, parsed_dir):
    """Extract Question, Answer User pairs and output to file.

    (1) Extract "Q" - "A"
        The list of AnswerOwnerList contains <aid>-<owner_id> pairs
        Format:
            <Qid> <Aid>
        E.g.
            100 1011
            21 490

    (2) Extract "Q" - Accepted answerer
        Format:
            <Qid> <Acc_Aid>
    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    # INPUT = "Record_Train.json"
    INPUT = "Record_All.json"

    OUTPUT_A = "Q_A.txt"
    OUTPUT_ACC = "Q_ACC.txt"

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT find {}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin, \
            open(parsed_dir + OUTPUT_A, "w") as fout_a, \
            open(parsed_dir + OUTPUT_ACC, "w") as fout_acc:
        for line in fin:
            data = json.loads(line)
            qid = data['QuestionId']
            aid_list = data['AnswererIdList']
            accaid = data['AcceptedAnswererId']
            for aid in aid_list:
                part_user.add(int(aid))
                print("{} {}".format(str(qid), str(aid)), file=fout_a)
            print("{} {}".format(str(qid), str(accaid)), file=fout_acc)


def extract_question_content(data_dir, parsed_dir):
    """Extract questions, content pairs from question file

    Question content pair format:
        <qid> <content>
    We extract both with and without stop-word version
        which is signified by "_nsw"

    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    INPUT = "Posts_Q.json"
    OUTPUT_T = "Q_title.txt"  # Question title
    OUTPUT_T_NSW = "Q_title_nsw.txt"  # Question title, no stop word
    OUTPUT_C = "Q_content.txt"  # Question content
    OUTPUT_C_NSW = "Q_content_nsw.txt"  # Question content, no stop word

    logger = logging.getLogger(__name__)

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT locate {}".format(data_dir + INPUT))

    sw_set = set(stopwords.words('english'))  # Create the stop word set


    # We will try both with or without stopwords to
    # check out the performance.
    with open(data_dir + INPUT, "r") as fin, \
            open(parsed_dir + OUTPUT_T, "w") as fout_t, \
            open(parsed_dir + OUTPUT_T_NSW, "w") as fout_t_nsw, \
            open(parsed_dir + OUTPUT_C, "w") as fout_c, \
            open(parsed_dir + OUTPUT_C_NSW, "w") as fout_c_nsw:

        for line in fin:
            data = json.loads(line)
            try:
                qid = data.get('Id')
                if qid not in qa_map:
                    continue
                title = data.get('Title')
                content = data.get('Body')

                # content, title = clean_str2(content), clean_str2(title)
                content = content.replace('\n', ' ').replace('\r', '')
                content_nsw = remove_stopwords(content, sw_set)
                title_nsw = remove_stopwords(title, sw_set)

                print("{} {}".format(qid, content_nsw),
                      file=fout_c_nsw)  # Without stopword
                print("{} {}".format(qid, content),
                      file=fout_c)  # With stopword
                print("{} {}".format(qid, title_nsw),
                      file=fout_t_nsw)  # Without stopword
                print("{} {}".format(qid, title),
                      file=fout_t)  # With stopword
            except:
                logger.info(e)
                logger.info("Error at Extracting question content and title: "
                            + str(data))
                continue
 


def extract_question_encoding(data_dir, parsed_dir):
    """Extract questions, content pairs from question file

    Question content pair format:
        <qid> <content>
    We extract both with and without stop-word version
        which is signified by "_nsw"

    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    INPUT = "Q_title.txt"
    OUTPUT_E = "Q_encoding.npy" #Question title encoding

    # Load the Sentence-BERT model
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    logger = logging.getLogger(__name__)

    if not os.path.exists(parsed_dir + INPUT):
        IOError("Can NOT locate {}".format(data_dir + INPUT))

    encodings = dict()
    qids = []
    titles = []
    with open(parsed_dir + INPUT, "r") as fin:
        for line in fin:
            try:
                qid, title = line.split(maxsplit=1)
            except:
                continue
            qids.append(qid)
            titles.append(title)

    title_encodings = model.encode(titles)
    encodings = dict(zip(qids, title_encodings))
    np.save(parsed_dir + OUTPUT_E, encodings)


def extract_answer_content(data_dir, parsed_dir):
    """Extract answeres, content pairs from answer file

    Answer content pair format:
        <answer id> <content>
    We extract both with and without stop-word version
        which is signified by "_nsw"

    Args:
        data_dir - data directory
        parsed_dir - parsed file directory
    """
    INPUT = "Posts_A.json"
    OUTPUT_C = "A_content.txt"  # Answer content
    OUTPUT_C_NSW = "A_content_nsw.txt"  # Answer content, no stop word

    logger = logging.getLogger(__name__)

    if not os.path.exists(data_dir + INPUT):
        IOError("Can NOT locate {}".format(data_dir + INPUT))

    sw_set = set(stopwords.words('english'))  # Create the stop word set


    # We will try both with or without stopwords to
    # check out the performance.
    with open(data_dir + INPUT, "r") as fin, \
            open(parsed_dir + OUTPUT_C, "w") as fout_c, \
            open(parsed_dir + OUTPUT_C_NSW, "w") as fout_c_nsw:
        for line in fin:
            data = json.loads(line)
            try:
                answer_id = data.get('Id', None)
                aid = data.get('OwnerUserId', None)
                qid = data.get('ParentId', None)
                entry = qa_map.get(qid, None)
                if not entry:
                    continue
                if (aid, answer_id) not in entry["AnswererAnswerTuples"]:
                    continue

                content = data.get('Body')

                content = clean_str2(content)
                content_nsw = remove_stopwords(content, sw_set)

                print("{} {}".format(answer_id, content_nsw),
                      file=fout_c_nsw)  # Without stopword
                print("{} {}".format(answer_id, content),
                      file=fout_c)  # With stopword

            except:
                logger.info(e)
                logger.info("Error at Extracting answer content: "
                            + str(data))
                continue
 

def answerer_stats_2(parsed_dir):
    INPUT = "Q_A.txt"
    OUTPUT = "answerer_stats_2.json"

    if not os.path.exists(parsed_dir + INPUT):
        IOError("Can not locate {}".format(parsed_dir + INPUT))


    aid_qid_list = dict()
    with open(parsed_dir + INPUT, "r") as fin:
        for line in fin:
            qid, aid = line.split()
            if aid not in  aid_qid_list:
                aid_qid_list[aid] = []

            aid_qid_list[aid].append(qid)


    with open(parsed_dir + OUTPUT, "w") as fout:
        fout.write(json.dumps(aid_qid_list))
   

def extract_answer_score(data_dir, parsed_dir):
    """Extract the answers vote, a.k.a. Scores.

    This information might be useful when
        the accepted answer is not selected.

    Args:
        data_dir - Input data dir
        parsed_dir - Output data dir
    """
    INPUT = "Posts_A.json"
    OUTPUT = "A_score.txt"

    logger = logging.getLogger(__name__)

    if not os.path.exists(data_dir + INPUT):
        IOError("Cannot find file{}".format(data_dir + INPUT))

    with open(data_dir + INPUT, "r") as fin, \
        open(parsed_dir + OUTPUT, "w") as fout:
        for line in fin:
            data = json.loads(line)
            try:
                aid = data.get('Id')
                score = data.get('Score')
                print("{} {}".format(aid, score), file=fout)
            except:
                logging.info("Error at Extracting answer score: "
                             + str(data))
                continue


def extract_question_best_answerer(data_dir, parsed_dir):
    """Extract the question-best-answerer relation

    Args:
        data_dir  - as usual
        parsed_dir  -  as usual
    """
    INPUT_A = "Posts_A.json"
    INPUT_MAP = "Record_Train.json"
    OUTPUT = "Q_ACC_A.txt"

    if not os.path.exists(data_dir + INPUT_A):
        IOError("Cannot find file {}".format(data_dir + INPUT_A))
    if not os.path.exists(data_dir + INPUT_MAP):
        IOError("Cannot find file {}".format(data_dir + INPUT_MAP))

    accanswerid_uaid = {}  # Accepted answer id to Answering user id
    answerid_score = {}  # Answer id to answer scores
    with open(data_dir + INPUT_A, "r") as fin_a, \
        open(data_dir + INPUT_MAP, "r") as fin_map, \
        open(parsed_dir + OUTPUT, "w") as fout:

        # build acc-a dict
        for line in fin_a:
            data = json.loads(line)
            try:
                answerid = data.get("Id")
                if answerid == "5":
                    print(100)
                score = data.get("Score")
                uaid = data.get("OwnerUserId")
                answerid_score[answerid] = score
                accanswerid_uaid[answerid] = uaid  # uaid is rid
            except:
                logging.info(
                    "Error at Extracting question, best answer user: "
                    + str(data))

        print(len(accanswerid_uaid))
        for line in fin_map:
            data = json.loads(line)
            try:
                qid = data.get('QuestionId')
                if "AcceptedAnswerID" in data:  # If acc answer exists
                    acc_answerid = data.get('AcceptedAnswerId')
                else:
                # If acc answer doesn't exist, choose highest score answer
                    ans = data.get('AnswerIdList')
                    ans = list(zip(*ans))[0]
                    scores = [answerid_score[answerid] for answerid in ans]
                    max_ind = scores.index(max(scores))
                    acc_answerid = ans[max_ind]
                uaccid = accanswerid_uaid[acc_aid]
                print("{} {}".format(qid, uaccid), file=fout)
            except:
                print(1)
                logging.info(
                    "Error at Extracting question, best answer user: "
                     + str(data))


def extract_question_best_answerer_2(data_dir, parsed_dir):
    """Extract the question-best-answerer relation

    Args:
        data_dir  - as usual
        parsed_dir  -  as usual
    """
    INPUT_A = "Posts_A.json"
    # INPUT_MAP = "Record_Train.json"
    # Uncomment this when running NeRank
    INPUT_MAP = "Record_All.json"
    OUTPUT = "Q_ACC_A.txt"

    if not os.path.exists(data_dir + INPUT_A):
        IOError("Cannot find file {}".format(data_dir + INPUT_A))
    if not os.path.exists(data_dir + INPUT_MAP):
        IOError("Cannot find file {}".format(data_dir + INPUT_MAP))

    accanswerid_uaid = {}  # Accepted answer id to Answering user id
    answerid_score = {}  # Answer id to answer scores
    with open(data_dir + INPUT_MAP, "r") as fin_map, \
        open(parsed_dir + OUTPUT, "w") as fout:

        for line in fin_map:
            data = json.loads(line)
            try:
                qid = data.get('QuestionId')
                acc_aid = data.get("AcceptedAnswererId")
                if qid and acc_aid:
                    print("{} {}".format(qid, acc_aid), file=fout)
            except:
                print(1)
                logging.info(
                    "Error at Extracting question, best answer user: "
                     + str(data))


def write_part_users(parsed_dir):
    OUTPUT = "QA_ID.txt"
    with open(parsed_dir + OUTPUT, "w") as fout:
        IdList = list(part_user)
        IdList.sort()
        for index, user_id in enumerate(IdList):
            print("{} {}".format(index + 1, user_id), file=fout)


def preprocess_(dataset, threshold, prop_test, sample_size):
    DATASET = dataset
    RAW_DIR = os.getcwd() + "/raw/{}/".format(DATASET)
    DATA_DIR= os.getcwd() + "/data/{}/".format(DATASET)
    PARSED_DIR = os.getcwd() + "/data/parsed/{}/".format(DATASET)
    print(RAW_DIR)
    print(PARSED_DIR)
    print("Preprocessing {} ...".format(dataset))

    if not os.path.exists(RAW_DIR):
        print("{} dir or path doesn't exist.\n"
              "Please download the raw data set into the /raw."
              .format(RAW_DIR), file=sys.stderr)
        sys.exit()

    if not os.path.exists(DATA_DIR):
        print("{} data dir not found.\n"
              " Creating a folder for that."
              .format(DATA_DIR))
        os.makedirs(DATA_DIR)

    if not os.path.exists(PARSED_DIR):
        print("{} dir or path NOT found.\n"
              "Creating a folder for that."
              .format(PARSED_DIR))
        os.makedirs(PARSED_DIR)

    if os.path.exists(DATA_DIR + "log.log"):
        os.remove(DATA_DIR + "log.log")

    # Setting up loggers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_fh = logging.FileHandler(DATA_DIR + "log.log")
    log_fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_fh.setFormatter(formatter)
    logger.addHandler(log_fh)

    # Split contest to question and answer
    print("\tSpliting post")
    split_post(raw_dir=RAW_DIR, data_dir=DATA_DIR)

    # Extract question-user, answer-user, and question-answer information
    # Generate Question and Answer/User map
    print("\tProcessing QA")
    process_QA(data_dir=DATA_DIR)

    print("\tGenerating question statistics...")
    question_stats(data_dir=DATA_DIR)

    print("\tExtracting question content ...")
    extract_question_content(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)
    print("\tExtracting question encoding ...")

    extract_question_encoding(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)


    print("\tExtracting answer content ...")
    extract_answer_content(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)

    # print("\tBuilding test sets")
    # build_test_set(data_dir=DATA_DIR, parsed_dir=PARSED_DIR,
    #                threshold=threshold, test_sample_size=sample_size,
    #                test_proportion=prop_test)

    print("\tExtracting Q, R, A relations ...")
    extract_question_user(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)

    extract_question_tags(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)

    extract_question_answer_user(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)


    extract_answer_score(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)
    extract_question_best_answerer_2(data_dir=DATA_DIR, parsed_dir=PARSED_DIR)

    write_part_users(parsed_dir=PARSED_DIR)

    answerer_stats(data_dir=DATA_DIR)

    answerer_stats_2(parsed_dir=PARSED_DIR)

    print("\tBuilding test sets")
    build_test_set(data_dir=DATA_DIR, parsed_dir=PARSED_DIR,
                   threshold=threshold, test_sample_size=sample_size,
                   test_proportion=prop_test)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 3 + 1:
        print("\t Usage: {} [name of dataset] [threshold] [prop of test] [test sample size]"
              .format(sys.argv[0]), file=sys.stderr)
        sys.exit(0)
    threshold = int(sys.argv[2])
    test_proportion = float(sys.argv[3])
    sample_size = int(sys.argv[4])
    print(sys.argv[1])
    preprocess_(sys.argv[1], threshold, test_proportion, sample_size)
