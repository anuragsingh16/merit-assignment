import tensorflow as tf
import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import pickle

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load the dataset from Hugging Face (using "opus_books", "en-es")
dataset = load_dataset("opus_books", "en-es")

# Step 1: Data Preprocessing
# Get English and Spanish sentence pairs
english_sentences = [entry['en'] for entry in dataset['train']['translation']]
spanish_sentences = [entry['es'] for entry in dataset['train']['translation']]

# Tokenizing the sentences
def tokenize_sentences(sentences):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token='<UNK>')
    tokenizer.fit_on_texts(['<START> ' + s + ' <END>' for s in sentences])
    sequences = tokenizer.texts_to_sequences(['<START> ' + s + ' <END>' for s in sentences])
    return sequences, tokenizer

# Tokenize English and Spanish sentences
english_sequences, english_tokenizer = tokenize_sentences(english_sentences)
spanish_sequences, spanish_tokenizer = tokenize_sentences(spanish_sentences)

# Save tokenizers for inference
with open('english_tokenizer.pkl', 'wb') as f:
    pickle.dump(english_tokenizer, f)
with open('spanish_tokenizer.pkl', 'wb') as f:
    pickle.dump(spanish_tokenizer, f)

# Define constants for padding
MAX_INPUT_LENGTH = max(len(seq) for seq in english_sequences)
MAX_OUTPUT_LENGTH = max(len(seq) for seq in spanish_sequences)

# Padding sequences (post-padding)
english_sequences = pad_sequences(english_sequences, maxlen=MAX_INPUT_LENGTH, padding='post')
spanish_sequences = pad_sequences(spanish_sequences, maxlen=MAX_OUTPUT_LENGTH, padding='post')

# Step 2: Prepare Decoder Input and Output for Teacher Forcing
decoder_input_data = spanish_sequences[:, :-1]
decoder_output_data = spanish_sequences[:, 1:]

# Step 3: Train-Test Split
X_train, X_val, dec_input_train, dec_input_val, y_train, y_val = train_test_split(
    english_sequences, decoder_input_data, decoder_output_data, test_size=0.2, random_state=42
)

# Step 4: Build the Seq2Seq Model with Attention
def build_seq2seq_model(input_vocab_size, output_vocab_size, embedding_dim, hidden_units, max_input_len, max_output_len):
    encoder_input = layers.Input(shape=(max_input_len,))
    encoder_embedding = layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dim)(encoder_input)
    encoder_lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_input = layers.Input(shape=(max_output_len-1,))
    decoder_embedding = layers.Embedding(input_dim=output_vocab_size, output_dim=embedding_dim)(decoder_input)
    decoder_lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    attention = layers.Attention()([decoder_lstm_output, encoder_output])
    attention_output = layers.Concatenate(axis=-1)([decoder_lstm_output, attention])

    decoder_dense = layers.Dense(output_vocab_size, activation='softmax')
    decoder_output = decoder_dense(attention_output)

    model = models.Model([encoder_input, decoder_input], decoder_output)
    return model

# Hyperparameters
embedding_dim = 256
hidden_units = 512
batch_size = 64
epochs = 1  # Reduced for faster testing; increase for better results
input_vocab_size = len(english_tokenizer.word_index) + 1
output_vocab_size = len(spanish_tokenizer.word_index) + 1

# Build and compile the model
model = build_seq2seq_model(
    input_vocab_size, output_vocab_size, embedding_dim, hidden_units, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Step 5: Train the Model
model.fit(
    [X_train, dec_input_train], y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([X_val, dec_input_val], y_val)
)

# Step 6: Inference Setup (Encoder and Decoder Models)
encoder_model = models.Model(encoder_input, [encoder_output] + encoder_states)

# Decoder model for inference
decoder_state_input_h = layers.Input(shape=(hidden_units,))
decoder_state_input_c = layers.Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm_output, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
attention = layers.Attention()([decoder_lstm_output, encoder_output])
attention_output = layers.Concatenate(axis=-1)([decoder_lstm_output, attention])
decoder_output = decoder_dense(attention_output)
decoder_model = models.Model([decoder_input, encoder_output] + decoder_states_inputs, [decoder_output] + decoder_states)

# Step 7: Beam Search for Inference
def beam_search_decode(encoder_model, decoder_model, input_seq, spanish_tokenizer, beam_width=3, max_len=MAX_OUTPUT_LENGTH):
    enc_output, state_h, state_c = encoder_model.predict(input_seq, verbose=0)
    states = [state_h, state_c]

    sequences = [[[], 0.0, states]]  # [token_list, score, states]
    start_token = spanish_tokenizer.word_index['<START>']

    for _ in range(max_len):
        all_candidates = []
        for seq, score, states in sequences:
            if seq and seq[-1] == spanish_tokenizer.word_index.get('<END>', 0):
                all_candidates.append([seq, score, states])
                continue
            dec_input = np.array([[seq[-1] if seq else start_token]])
            dec_input = pad_sequences(dec_input, maxlen=MAX_OUTPUT_LENGTH-1, padding='post')
            output_tokens, h, c = decoder_model.predict([dec_input, enc_output] + states, verbose=0)
            new_states = [h, c]
            top_probs = np.argsort(output_tokens[0, -1, :])[-beam_width:]
            top_scores = np.log(output_tokens[0, -1, top_probs])
            for prob, score_inc in zip(top_probs, top_scores):
                new_seq = seq + [prob]
                all_candidates.append([new_seq, score + score_inc, new_states])
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = sequences[0][0]
    return [spanish_tokenizer.index_word.get(idx, '<UNK>') for idx in best_seq]

# Step 8: BLEU Score Evaluation
def evaluate_bleu(model, encoder_model, decoder_model, input_data, target_sentences, spanish_tokenizer):
    predictions = []
    for input_seq in input_data:
        input_seq = input_seq.reshape(1, -1)
        pred_tokens = beam_search_decode(encoder_model, decoder_model, input_seq, spanish_tokenizer)
        predictions.append(' '.join(pred_tokens))

    references = [[t.split()] for t in target_sentences]
    candidates = [pred.split() for pred in predictions]

    bleu_score = corpus_bleu(references, candidates)
    return bleu_score

# Evaluate on validation set
val_indices = np.random.choice(len(X_val), 100, replace=False)
X_val_subset = X_val[val_indices]
val_sentences_subset = np.array(spanish_sentences)[val_indices]
bleu_score = evaluate_bleu(model, encoder_model, decoder_model, X_val_subset, val_sentences_subset, spanish_tokenizer)
print(f"BLEU Score: {bleu_score:.4f}")

# Save the models
model.save('seq2seq_model.h5')
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
