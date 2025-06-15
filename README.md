# Deception Detection In Diplomacy
## 🧠 Introduction

**Diplomacy** is a strategy board game set in pre-WWI Europe where seven players represent major powers and compete for territorial control. Unlike most games, there are no dice rolls or hidden cards—victory depends entirely on negotiation and alliances between players. However, these alliances are non-binding, making **deception and betrayal core mechanics of gameplay**.

In this project, we tackle the task of **deception detection**: given a message exchanged between players, can we determine if the sender intended to lie? This is especially challenging because:

- Only ~5% of messages are deceptive
- Lies are often subtle, strategic, or disguised as cooperation
- Context across multiple turns is critical

We use the **QANTA Diplomacy Deception dataset**, which provides labeled messages annotated by the sender (actual intent). Our models aim to detect lies by modeling message content, player dynamics, historical dialogue, and in-game context.

## 🧪 Contributions

**Both our proposed models outperform baselines from prior work and highlight the importance of integrating social structure and historical context for deception detection.**

We propose two complementary models to detect deception in Diplomacy conversations:

- **LieDetectorGAT**:  
  A graph-based model that represents each game as a dynamic interaction graph.  
  - Nodes = players  
  - Edges = messages enriched with BERT embeddings, deception lexicon cues, and strategic metadata  
  - Uses Graph Attention (GATv2) to capture the importance of both **who** is speaking and **what** is said in context.

- **HiS-Attention (Historical-Structured Attention)**:  
  A multimodal transformer-based model that fuses:  
  - The current message (BERT-encoded)  
  - Game state features (e.g., season, score change)  
  - Dialogue history (last 10 messages)  
  These streams are integrated via **learnable query attention pooling** to produce deception-aware message representations.

Both models address key limitations in prior work—lack of context, sparse deception signals, and weak inter-player modeling—by capturing the **linguistic, relational, and strategic structure** of deception.

## 📦 Dataset

We use the **QANTA Diplomacy Deception Dataset**, derived from real games of Diplomacy and annotated by both message senders and receivers. For this project, we focus solely on the **sender's actual intent**—whether the message was truthful or deceptive.

Each data sample includes:
- The message text
- Sender and receiver identities
- Turn-specific game metadata (season, year, power dynamics)
- Historical context of prior communication (optional)

The dataset is **highly imbalanced**, with only ~5% of messages labeled as deceptive:

| Split      | Truthful | Deceptive |
|------------|----------|-----------|
| Train      | 12,541   | 591       |
| Validation | 1,360    | 56        |
| Test       | 2,501    | 240       |

This imbalance requires specialized architectures and training strategies to ensure robust detection of minority-class (lie) instances.

## 🕸️ LieDetectorGAT

**LieDetectorGAT** is a graph-based neural network that models the social dynamics of Diplomacy. Each game is treated as a directed graph, where players are nodes and messages are edges. The goal is to predict if a message (edge) is deceptive, using not only its content but also strategic context and inter-player relationships.

<img width="479" alt="image" src="https://github.com/user-attachments/assets/905f1de9-a4ef-46e3-b7ab-6ff93d983113" />


### 🔩 Step-by-Step Architecture

1. **Graph Construction**  
   - Each player in a game is a **node** (7 total).
   - Each message is a **directed edge** from sender to receiver.
   - Messages are ordered chronologically to prevent future info leakage.

2. **Edge Feature Extraction**  
   For each message, we compute a 783-dimensional edge vector:
   - **BERT [CLS] embedding** (768D)
   - **Deception lexicon features** (10D): counts of deceptive words
   - **Strategic metadata**:
     - Running lie count (1D)
     - Average power score delta (1D)
     - Game turn info (season, year bucket, message index) (3D)

3. **Edge Encoder**  
   - A feedforward MLP transforms edge features into learned embeddings.

4. **Graph Attention Layers**  
   - We use **GATv2Conv** to propagate information across the graph.
   - Attention is modulated by edge features to prioritize deceptive cues.

5. **Deception Classification**  
   - For each edge (message), we use a 2-layer MLP with:
     - Sender node embedding
     - Receiver node embedding
     - Edge embedding  
   - Output: scalar logit → sigmoid → deception probability

### ⚙️ Training Setup

- Optimizer: **AdamW**, learning rate: `3e-4`
- Loss: **BCEWithLogitsLoss** with class weights to address imbalance
- Batch size: 1 game per batch (preserves graph structure)
- Dataset is sorted to maintain message chronology
- Framework: **PyTorch Geometric**

## 🔀 HiS-Attention (Historical-Structured Attention)

**HiS-Attention** is a transformer-based multimodal architecture designed to capture the full context behind a message — not just its content, but the surrounding game state and conversation history. Unlike traditional models that concatenate features, HiS-Attention uses **cross-modal attention** to fuse information dynamically and contextually.

<img width="534" alt="image" src="https://github.com/user-attachments/assets/12a0700a-9868-4340-8d50-63314a502cdc" />


### 🧩 Step-by-Step Architecture

1. **Message Encoding**  
   - The current message is encoded using `BERT-base-uncased` to extract a contextual embedding.

2. **Game State Encoding**  
   - Structured metadata (season, year, score delta, etc.) is processed via a small feedforward neural network.
   - Categorical features are one-hot encoded; numerical features are used as-is.

3. **Dialogue History Encoding**  
   - The last **10 messages** from the same sender to the same receiver are collected.
   - Each is passed through BERT, then aggregated using a **Transformer encoder** with positional encodings to capture conversational flow.

4. **Fusion via Query-Based Attention Pooling**  
   - The outputs from message, game state, and history encoders are treated as a 3-element sequence.
   - **Multi-head self-attention** allows interactions between these modalities.
   - A **learnable query vector** attends over the sequence to extract a fused representation.

5. **Classification Head**  
   - The pooled representation is passed to a feedforward network to output a deception probability.

### ⚙️ Training Setup

- Optimizer: **Adam**, learning rate: `1e-5`
- Loss: **Weighted Binary Cross-Entropy** (positive class weight ≈ 21.2)
- Batch size: `16`
- Trained for `10` epochs with learning rate scheduling and gradient clipping

## 📊 Results

We evaluate both models—HiS-Attention and LieDetectorGAT—on a held-out test set using Macro F1, Lie F1, and Accuracy as our primary metrics.


<img width="423" alt="image" src="https://github.com/user-attachments/assets/69d3b173-3897-4786-858d-a83a86162b03" />

### 🔍 Analysis

- **HiS-Attention** offers strong overall performance with rich multimodal modeling, but struggles with **low recall** on deceptive messages (Lie F1 = 22%).
- **LieDetectorGAT** achieves better **deception recall** (38.8%) and Lie F1 (26.6%), indicating stronger capability to identify lies, though with slightly lower precision.
- Speaker-level analysis reveals that players like **Germany** have higher misclassification rates (32.3%) due to more complex or ambiguous rhetoric, while **Italy** had the lowest error rate (~5%).

Both models outperform baselines from prior work and highlight the importance of integrating social structure and historical context for deception detection.


## 📚 Citation

If you use this code or build upon our models, please cite the original dataset paper:

```bibtex
@inproceedings{peskov2020it,
  title={It Takes Two to Lie: One to Lie, and One to Listen},
  author={Peskov, Denis and Cheng, Benny and Elgohary, Ahmed and Barrow, Joe and Danescu-Niculescu-Mizil, Cristian and Boyd-Graber, Jordan},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3811--3824},
  year={2020}
}

