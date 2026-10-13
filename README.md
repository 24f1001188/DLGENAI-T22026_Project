# Smart MCQ Solver: NLP, Deep Learning & Retrieval-Augmented Ranking

**A comparative study of NLP, deep learning, transformer models, and Retrieval-Augmented Generation (RAG) for multiple-choice question answering.**

## Overview

Smart MCQ Solver is a Natural Language Processing (NLP) and deep learning project developed for a Kaggle competition that requires ranking candidate answers for multiple-choice questions.

Given a question and five candidate answers (A–E), the objective is to rank the options and predict the **top three most likely answers**. The competition evaluates predictions using **Mean Average Precision at 3 (MAP@3)**.

The project follows a progressive experimentation approach, starting with traditional lexical similarity methods and advancing to semantic embeddings, zero-shot natural language inference, retrieval-augmented ranking, supervised deep learning, and transformer fine-tuning. A weighted ensemble of complementary models was also evaluated.

The goal was to compare these approaches within a common evaluation framework and investigate how different text representations and learning strategies affect MCQ ranking performance.

## Key Highlights

- Implemented and compared **8 approaches**, from simple baselines to deep learning and ensemble methods.
- Developed a custom **Bidirectional LSTM (BiLSTM)** model using TensorFlow and Keras.
- Fine-tuned **DistilBERT** for binary question–option classification.
- Built a **Retrieval-Augmented Generation (RAG) pipeline** using Sentence Transformers, FAISS, and cross-encoder reranking.
- Evaluated zero-shot natural language inference using BART-MNLI.
- Removed exact duplicate MCQs and used a group-aware train–validation split to reduce data leakage.
- Used **MAP@3** as the primary metric for evaluating ranked predictions.
- Tracked training behaviour using Weights & Biases (W&B).
- Achieved a **public Kaggle MAP@3 score of 0.75353**, with a public leaderboard rank of 592.

## Problem Statement

The competition dataset contains multiple-choice questions with five candidate answers. Each training question includes its correct answer, while the test questions do not have answer labels.

Rather than predicting a single answer directly, the task requires assigning a score to each of the five options, ranking them in descending order, and submitting the top three.

This problem involves several challenges:

- Understanding the relationship between a question and its candidate answers.
- Capturing semantic meaning beyond exact word overlap.
- Learning contextual relationships between questions and options.
- Leveraging relevant information retrieved from training data.
- Handling class imbalance when converting each MCQ into binary question–option examples.
- Generalizing to unseen competition data.

## Dataset

The project used the competition's training and test datasets.

| Dataset characteristic | Count |
|---|---:|
| Original training MCQs | 2,000 |
| Exact duplicates removed | 183 |
| Unique training MCQs | 1,817 |
| Training MCQs after splitting | 1,456 |
| Validation MCQs | 361 |
| Competition test MCQs | 500 |

Each MCQ consists of a question prompt, five candidate answer options (A–E), and the correct answer label in the training data.

### Data Preprocessing

The following preprocessing and validation steps were performed:

1. **Duplicate removal:** Removed 183 exact duplicate MCQs by comparing the complete question and its five candidate options.
2. **Text normalization:** Converted text to lowercase and normalized unnecessary or repeated whitespace.
3. **Instruction-phrase removal:** Removed repeated instructional phrases, such as "pick the best possible answer" and "among the listed options," from the BiLSTM question prompts.
4. **Group-aware splitting:** Used `GroupShuffleSplit` with the question prompt as the grouping variable, a 20% validation fraction, and random seed 42.
5. **Leakage checks:** Verified that the training and validation partitions had zero exact prompt overlaps, zero full-MCQ overlaps, and zero shared sample IDs.
6. **Sequence length selection:** Chose a maximum sequence length of 128 based on exploratory analysis of question–option text lengths.

No synthetic data augmentation was applied.

## Methodology and Model Development

The project compared lexical, semantic, retrieval-based, and supervised learning approaches. Each method generated scores for the five candidate answers, which were then ranked for MAP@3 evaluation.

### 1. Majority-Class Baseline

A simple baseline was implemented by always predicting the most frequent answer label observed in the training data.

This provided a reference point for evaluating the more sophisticated approaches.

**Validation MAP@3:** 0.3906

### 2. TF-IDF Cosine Similarity

TF-IDF was used to represent question and candidate-option text through weighted word-occurrence features. Cosine similarity was then used to rank the candidate answers.

This approach served as a traditional lexical baseline but was limited by its reliance on word overlap.

**Validation MAP@3:** 0.2955

### 3. Sentence Transformer Similarity

Sentence Transformer embeddings were used to represent questions and candidate answers as dense semantic vectors.

Cosine similarity was used to compare their representations, allowing the model to capture semantic relationships beyond exact word matching.

**Validation MAP@3:** 0.4155

### 4. Zero-Shot BART-MNLI

The pretrained `facebook/bart-large-mnli` model was evaluated without task-specific fine-tuning.

Each question–option pair was processed using the natural language inference model, and the resulting scores were used to rank the five candidate answers.

This approach explored whether a pretrained model could assess question–answer compatibility without being explicitly trained on the competition's labelled examples.

**Validation MAP@3:** 0.5471

### 5. Retrieval-Augmented Generation (RAG)

A retrieval-augmented pipeline was developed to incorporate relevant information retrieved from a knowledge base derived from the training data.

**Pipeline components:**

- **Sentence Transformers:** Generated dense semantic representations for retrieval.
- **FAISS:** Enabled efficient vector similarity search.
- **Cross-encoder:** Reranked retrieved information based on relevance.
- **Retrieved context:** Combined with the question and candidate options before the final scoring stage.

The retrieved information provided additional context for evaluating candidate answers instead of relying exclusively on direct question–option similarity.

**Validation MAP@3:** 0.8850

### 6. Custom BiLSTM Model

A Bidirectional Long Short-Term Memory (BiLSTM) network was developed from scratch using TensorFlow and Keras.

The model learned from labelled question–option pairs, providing a supervised alternative to the lexical, semantic, and zero-shot approaches.

#### Question–Option Formulation

Each MCQ was converted into five question–option pairs:

`Question [SEP] Candidate Option`

The correct option was labelled `1`, while the four incorrect options were labelled `0`.

The model performed binary classification for each pair. During inference, it generated a score for every candidate answer, sorted the five scores, and selected the top three options.

#### Architecture

The model consisted of the following layers:

1. **TextVectorization:** Converted text into integer sequences using a vocabulary of 3,500 tokens.
2. **Embedding:** Generated dense word representations with an embedding dimension of 64.
3. **Bidirectional LSTM:** Used 64 units to learn sequential information from both directions.
4. **Global Max Pooling:** Retained the strongest learned features across the sequence.
5. **Dropout:** Applied a dropout rate of 0.30 to help reduce overfitting.
6. **Dense layer:** Used 32 units with ReLU activation.
7. **Output layer:** Used one sigmoid unit to produce a question–option score.

The maximum sequence length was 128 tokens.

#### Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Binary cross-entropy |
| Batch size | 32 |
| Maximum epochs | 10 |
| Early stopping patience | 2 epochs |
| Checkpointing | Best validation-loss weights |

Early stopping monitored validation loss, restored the best weights, and helped limit unnecessary training.

**Validation MAP@3:** 1.0000

### 7. Fine-Tuned DistilBERT

A pretrained `distilbert-base-uncased` transformer was fine-tuned for binary question–option classification.

Unlike the custom BiLSTM, which learned its word representations from the competition data, DistilBERT started with pretrained transformer representations and adapted them to the MCQ ranking task.

#### Input Representation

Each input was formed as:

`Question [SEP] Candidate Option`

The Hugging Face tokenizer converted the text into token IDs and attention masks.

- Maximum sequence length: 128 tokens
- Padding: Fixed-length padding
- Truncation: Enabled
- Output labels: Two classes, representing correct and incorrect options

#### Training Configuration

| Parameter | Configuration |
|---|---|
| Pretrained model | `distilbert-base-uncased` |
| Framework | Hugging Face Transformers |
| Training question–option pairs | 7,280 |
| Validation question–option pairs | 1,805 |
| Learning rate | 2e-5 |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Training epochs | 3 |
| Weight decay | 0.01 |
| Loss | Weighted cross-entropy |
| Checkpoint selection | Best model |

The model was fine-tuned end-to-end rather than used solely as a fixed feature extractor.

Because each question produced one positive pair and four negative pairs, the binary classification dataset was imbalanced. A custom `WeightedTrainer` was implemented with class weights of 0.625 for incorrect options and 2.5 for correct options.

#### Validation Results

| Metric | Score |
|---|---:|
| Validation loss | 0.2603 |
| Accuracy | 0.9335 |
| Precision | 0.8320 |
| Recall | 0.8366 |
| F1-score | 0.8343 |
| MAP@3 | 0.9718 |

The binary classification metrics evaluate individual question–option pairs, while MAP@3 evaluates the final ranking of the five options for each question.

### 8. Weighted Ensemble

A weighted ensemble combining the BiLSTM, DistilBERT, and RAG pipeline was also evaluated.

The purpose of the ensemble was to combine complementary approaches:

- **BiLSTM:** Supervised sequence-based learning.
- **DistilBERT:** Pretrained transformer-based contextual representations.
- **RAG:** Retrieval-based contextual information.

The ensemble achieved a validation MAP@3 of 1.0000, matching the highest validation score among the evaluated approaches.

However, the near-perfect validation results did not translate directly to the unseen Kaggle test set, highlighting the importance of evaluating generalization rather than relying exclusively on a single validation split.

## Results and Model Comparison

The primary evaluation metric was Mean Average Precision at 3 (MAP@3).

MAP@3 rewards placing the correct answer higher in the predicted ranking and evaluates whether it appears among the three submitted options.

### Validation Performance

| Model | Validation MAP@3 |
|---|---:|
| Majority-class baseline | 0.3906 |
| TF-IDF cosine similarity | 0.2955 |
| Sentence Transformer | 0.4155 |
| BART-MNLI zero-shot | 0.5471 |
| RAG pipeline | 0.8850 |
| Custom BiLSTM | 1.0000 |
| Fine-tuned DistilBERT | 0.9718 |
| Weighted ensemble | 1.0000 |

### Kaggle Competition Performance

| Metric | Result |
|---|---:|
| Public Kaggle MAP@3 | **0.75353** |
| Public leaderboard rank | **592** |

The results demonstrate substantial improvements from the early lexical and semantic baselines to the retrieval-based and supervised approaches on the local validation split.

The BiLSTM and weighted ensemble achieved the highest validation MAP@3, followed by DistilBERT. The RAG pipeline also performed strongly compared with the simpler baselines.

However, the difference between validation performance and the public leaderboard score showed that the selected validation split did not fully represent performance on unseen competition data. This is an important limitation of the experiments and a key lesson from the project.

## Experiment Tracking

Weights & Biases (W&B) was used to monitor training behaviour during the BiLSTM and DistilBERT experiments.

Training and validation loss histories were examined to understand optimization behaviour and investigate potential overfitting.

Monitoring loss progression alongside evaluation metrics provided a more complete view of model training than relying only on final scores.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| NLP and text preprocessing | TF-IDF, tokenization, text normalization |
| Semantic embeddings | Sentence Transformers |
| Zero-shot classification | BART-MNLI |
| Retrieval | FAISS |
| Reranking | Cross-encoder |
| Deep learning | TensorFlow, Keras |
| Recurrent neural network | Bidirectional LSTM (BiLSTM) |
| Transformer model | DistilBERT |
| Model training | Hugging Face Transformers |
| Experiment tracking | Weights & Biases |
| Evaluation | MAP@3, accuracy, precision, recall, F1-score |

## Key Learnings

This project provided practical experience in several areas of applied machine learning:

- **Comparative model evaluation:** Implementing multiple approaches and comparing them using a consistent ranking metric.
- **NLP representations:** Understanding the differences between lexical features, semantic embeddings, recurrent networks, and transformer representations.
- **Retrieval-augmented systems:** Integrating vector search and cross-encoder reranking into a question-answering workflow.
- **Question–option classification:** Transforming a ranking problem into binary classification over candidate answers.
- **Imbalanced learning:** Applying weighted cross-entropy to account for the one-positive-to-four-negative class distribution.
- **Data leakage prevention:** Removing exact duplicates and applying a group-aware train–validation split.
- **Experiment monitoring:** Using W&B to track training and validation losses.
- **Generalization:** Recognizing that strong local validation performance does not guarantee comparable performance on unseen data.

## Limitations and Future Improvements

The primary limitation was the gap between the near-perfect validation scores of some models and the final public Kaggle MAP@3 score of 0.75353.

Potential improvements include:

1. **Robust validation:** Evaluate models using multiple group-aware validation splits to obtain more reliable estimates of generalization.
2. **Hyperparameter tuning:** Systematically optimize model architecture, learning rates, sequence lengths, and training configurations.
3. **Improved retrieval:** Investigate alternative embedding models, retrieval strategies, and cross-encoder reranking configurations.
4. **Stronger pretrained models:** Evaluate other suitable pretrained language models for question–option ranking.
5. **Ensemble optimization:** Experiment with model weights and calibration strategies to improve the combination of BiLSTM, DistilBERT, and RAG scores.
6. **Generalization analysis:** Investigate why the local validation results differed from the public leaderboard results before selecting the final model.

## References

1. Hugging Face — DistilBERT Base, Uncased: https://huggingface.co/distilbert/distilbert-base-uncased
2. Hugging Face — BART Large MNLI: https://huggingface.co/facebook/bart-large-mnli
3. Sentence Transformers Documentation: https://www.sbert.net/
4. TensorFlow/Keras Documentation: https://www.tensorflow.org/
5. Weights & Biases Documentation: https://docs.wandb.ai/

## Author

**Hemalatha Baskar**

- B.Tech in Computer Science and Engineering — IIITDM Kancheepuram
- BS in Data Science and Applications — IIT Madras

Interested in Full-Stack Development, Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, and Generative AI, with a focus on building scalable software applications and intelligent solutions to real-world problems.

---

*This project was developed as part of a Deep Learning and Generative AI project and evaluated using the Smart MCQ Solver Kaggle competition.*
