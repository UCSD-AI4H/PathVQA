import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
from os.path import isfile, join
import utils
import re
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pylab


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_path', type=str)
	parser.add_argument('--model_path', type=str, default = 'jkooy/research/model3.ckpt',
                       help='Model Path')
	parser.add_argument('--num_lstm_layers', type=int, default=2)
	parser.add_argument('--cnn7_feature_length', type=int, default=512)
	parser.add_argument('--rnn_size', type=int)
	parser.add_argument('--embedding_size', type=int),
	parser.add_argument('--word_emb_dropout', type=float)
	parser.add_argument('--image_dropout', type=float)
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--question', type=str, help='Question')
	parser.add_argument('--lstm_direc', type=str)
	

	args = parser.parse_args()
    #Extract vocabulary of question and answer
	vocab_data = data_loader.get_question_answer_vocab(data_dir = args.data_dir)

    #Build q_map but seems useless, ans_map useful
	q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}
	ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
	cnn7_features = utils.extract_cnn7_features(args.image_path, join(args.data_dir, 'vgg16.tfmodel'))
	
    #Word Splitting
	question_vocab = vocab_data['question_vocab']
	word_regex = re.compile(r'\w+')
	question_words = re.findall(word_regex, args.question)
    #Find question's word in ques_vocab,record it in ques_id
	question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
	base = vocab_data['max_question_length'] - len(question_words)
	for i in range(0, len(question_words)):
		if question_words[i] in question_vocab:
			question_ids[0][base + i] = question_vocab[ question_words[i] ]
		else:
			question_ids[0][base + i] = question_vocab['UNK']

	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'cnn7_feature_length' : args.cnn7_feature_length,
		'lstm_steps' : vocab_data['max_question_length'] + 1,
		'q_vocab_size' : len(vocab_data['question_vocab']),
		'ans_vocab_size' : len(vocab_data['answer_vocab']),
	}
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_prediction, t_ans_probab = model.build_generator(batch=1)
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)
    #predict
	pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
        input_tensors['cnn7']:cnn7_features,
        input_tensors['sentence']:question_ids,
    })

	print("Image:", args.image_path)
	print("Question:", args.question)
	print("Ans:", ans_map[pred[0]])
	answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
	answer_probab_tuples.sort()
	print("Top Answers")
	sess.close()
	for i in range(5):
		print(ans_map[ answer_probab_tuples[i][1] ])

if __name__ == '__main__':
	main()
