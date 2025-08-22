flowchart TD
A[Input Audio: input_values] --> B[Wav2Vec2 Encoder]

B --> C[Hidden States<br/>Shape: (batch, seq_len, hidden_size)]

C --> D[AttentivePool<br/>Pooler]
C --> E[SpeakerAdversary Branch<br/>if adv_lambda > 0]

D --> F[Stats Projector<br/>2*hidden_size → hidden_size]
F --> G[Final Projector<br/>hidden_size → classifier_proj_size]
G --> H[Emotion Classifier<br/>→ num_labels]

E --> I[Mean Pooling<br/>seq_len → 1]
I --> J[Gradient Reversal Layer<br/>GRL with λ scaling]
J --> K[Speaker Classifier<br/>→ num_speakers]

H --> L[Emotion Logits]
K --> M[Speaker Logits<br/>Optional]

N[Labels] --> O{Labels & class_weights<br/>provided?}
O -->|Yes| P[CrossEntropyLoss<br/>with class_weights<br/>+ label_smoothing=0.1]
O -->|No| Q[No Emotion Loss]

R[Speaker IDs] --> S{adv_lambda > 0 &<br/>speaker_ids provided?}
S -->|Yes| T[Speaker Loss<br/>CrossEntropyLoss]
S -->|No| U[No Speaker Loss]

P --> V[Emotion Loss]
T --> W[Speaker Loss × λ]

V --> X[Total Loss = Emotion Loss + λ × Speaker Loss]
W --> X
Q --> Y[Loss = None]
U --> Y

X --> Z[Output: loss, emotion_logits]
Y --> Z

style A fill:#e1f5fe
style L fill:#c8e6c9
style M fill:#ffcdd2
style X fill:#fff3e0
style J fill:#f3e5f5
style P fill:#fff9c4
