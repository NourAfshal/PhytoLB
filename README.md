# EmoFlow: Modeling Dynamic Emotion Transitions in Conversations

## 📌 Overview

Human conversations are dynamic — emotions constantly evolve as people interact. A person may begin a discussion feeling calm, become excited, shift to frustration, and eventually feel relief. However, most existing emotion detection systems treat each message independently, ignoring the conversational context that shapes emotional transitions.

**EmoFlow** addresses this limitation by modeling how emotions change over time within conversations. Instead of simply predicting the emotion of a single message, our goal is to **predict the next emotion based on the full conversation history**.

---

## 🎯 Objective

The main objective of this project is to:

* Track emotional progression in conversations
* Model transitions between emotions
* Predict the next emotion in a dialogue sequence

---

## 📊 Dataset

We use the **GoEmotions dataset** (Demszky et al., 2020), developed by Google Research.

* ~58,000 Reddit comments
* 27 fine-grained emotion labels + neutral
* Includes emotions such as:

  * admiration
  * curiosity
  * excitement
  * grief
  * nervousness

Unlike traditional datasets with only a few basic emotions, GoEmotions provides a **rich and realistic representation of human feelings**. Additionally, since the data comes from Reddit threads, we can reconstruct full conversations and analyze emotional flow.

---

## ⚙️ Methodology

### 1. Conversation Reconstruction

* Extract Reddit threads from the dataset
* Rebuild conversations using parent-child relationships
* Represent each conversation as a sequence of messages with emotion labels

---

### 2. Emotion Transition Modeling

#### 🔹 Markov Chain (Baseline)

* Models probability of transitioning from one emotion to another:

  ```
  P(next_emotion | current_emotion)
  ```
* Simple and interpretable
* Captures basic emotional patterns

#### 🔹 LSTM Neural Network (Advanced)

* Uses sequential learning to capture long-term dependencies
* Input: sequence of previous emotions or text
* Output: predicted next emotion
* Handles complex conversational context

---

### 3. Emotion Flow Visualization

We create visual representations of emotional evolution:

* Timeline charts of conversations
* Emotion transition heatmaps
* Flow diagrams showing mood shifts

These visualizations help uncover patterns such as:

* Escalation to conflict
* Emotional recovery
* Stable vs volatile discussions

---

## 📈 Evaluation

We evaluate models using:

* Accuracy
* Top-k accuracy
* Comparison between Markov and LSTM performance

Key questions:

* Are emotional transitions predictable?
* Does context improve prediction?
* Which emotions lead to conflict or resolution?

---

## 💡 Applications

EmoFlow has several real-world applications:

* 🤖 Emotion-aware chatbots
* ⚠️ Early conflict detection in conversations
* 💬 Customer service sentiment monitoring
* 🧠 Mental health pattern analysis in online communities

---

## 📁 Project Structure

```
EmoFlow/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_markov_model.ipynb
│   ├── 3_lstm_model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── conversation_builder.py
│   ├── markov_model.py
│   ├── lstm_model.py
│   ├── visualization.py
│
├── results/
│   ├── plots/
│   ├── metrics/
│
├── README.md
├── requirements.txt
└── main.py
```

---

## 🚀 How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python main.py
```

---

## 📚 Reference

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
*GoEmotions: A Dataset of Fine-Grained Emotions.*
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

---

## 🧠 Key Insight

> Emotions are not isolated — they evolve. EmoFlow captures this evolution to better understand human communication.

---
