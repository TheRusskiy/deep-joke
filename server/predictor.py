import tensorflow as tf
import numpy as np
import pickle

class Predictor:
    def init(self):
        _, vocab_to_int, int_to_vocab = self.load_preprocess()
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.seq_length = 15
        self.symbol_lookup_table = {
            '.': '<DOT>',
            ',': '<COMMA>',
            ':': '<COLON>',
            ';': '<SEMICOLON>',
            '-': '<DASH>',
            '_': '<UNDERSCORE>',
            '!': '<EXCLAMATION>',
            '?': '<QUESTION>',
            '(': '<LEFTPARENTHESIS>',
            ')': '<RIGHTPARENTHESIS>'
        }
        self.loaded_graph = tf.Graph()
        save_dir = './save'
        self.session = tf.Session(graph=self.loaded_graph)
        # Load saved model
        with self.loaded_graph.as_default():
            loader = tf.train.import_meta_graph(save_dir + '.meta')
            loader.restore(self.session, save_dir)

    def predict(self, initial_words, break_on_end=True, mode='other'):
        gen_length = 50
        
        save_dir = './save'
        sess = self.session


        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs, keep_prob = self.get_tensors(self.loaded_graph)

        # Sentences generation setup
        gen_sentences = self.make_initial_sentences(initial_words.split(' '))
        prev_state = sess.run(initial_state, {input_text: np.array([[1]]), keep_prob: 1.0})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[self.vocab_to_int[word] for word in gen_sentences[-self.seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state, keep_prob: 1.0})
            
            pred_word = self.pick_word(probabilities[dyn_seq_length-1], mode)

            gen_sentences.append(pred_word)
        
        # Remove tokens
        the_joke = ' '.join(gen_sentences)
        the_end = the_joke.index('<EOS>')
        if the_end:
            while the_end and the_end < 30:
                the_joke = the_joke.replace('<EOS>', '', 1)
                the_end = the_joke.index('<EOS>')
            the_joke = the_joke[:the_end]
        for key, token in self.symbol_lookup_table.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            the_joke = the_joke.replace(' ' + token.lower(), key)
            the_joke = the_joke.replace(' ' + token, key)
            
        print(the_joke)
        return the_joke

    def load_preprocess(self):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        return pickle.load(open('./preprocess.p', mode='rb'))

    def get_tensors(self, loaded_graph):
        """
        Get input, initial state, final state, and probabilities tensor from <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
        """
        return loaded_graph.get_tensor_by_name("input:0"),\
               loaded_graph.get_tensor_by_name("initial_state:0"),\
               loaded_graph.get_tensor_by_name("final_state:0"),\
               loaded_graph.get_tensor_by_name("probs:0"),\
               loaded_graph.get_tensor_by_name("keep_prob:0")

    def pick_word(self, probabilities, mode):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :param int_to_vocab: Dictionary of word ids as the keys and words as the values
        :return: String of the predicted word
        """
        if mode == 'most_likely':
            choice = np.where(probabilities==max(probabilities))[0][0]
        else:
            choice = np.random.choice(len(probabilities), 1, p=probabilities)[0]
        return self.int_to_vocab[choice]

    def make_initial_sentences(self, words):
        results = []
        for word in words:
            if word in self.vocab_to_int:
                results.append(word)
            else:
                results.append('<UNKNOWN>')
        return results