# Multi-Source AI Tutor — Evaluation Results (N=100 Questions)

### Table 1: Overall System Performance Metrics
| Dimension | Metric | Score | 95% Confidence Interval |
|---|---|---|---|
| **Classification** | Accuracy | **36.0%** | [27.3%, 45.8%] |
| Classification | Macro Precision | 0.359 | - |
| Classification | Macro Recall | 0.378 | - |
| Classification | Macro F1 | **0.310** | - |
| **Lexical** | ROUGE-1 | 0.1158 ± 0.0369 | [0.1086, 0.1231] |
| Lexical | ROUGE-2 | 0.0336 ± 0.0169 | [0.0303, 0.0369] |
| Lexical | ROUGE-L | **0.0765 ± 0.0235** | [0.0719, 0.0811] |
| Lexical | BLEU | 0.0064 ± 0.0056 | [0.0053, 0.0075] |
| **Retrieval** | Avg Cosine Sim | 0.6779 | - |
| **LLM-as-Judge** | Accuracy | **88.8 ± 20.1 / 100** | [84.9, 92.7] |
| **LLM-as-Judge** | Faithfulness | **54.8 ± 35.2 / 100** | [47.7, 62.0] |
| **LLM-as-Judge** | Context Recall | **50.2 ± 33.4 / 100** | [40.4, 60.1] |
| **LLM-as-Judge** | Context Precision | **52.9 ± 32.4 / 100** | [44.3, 61.5] |
| **LLM-as-Judge** | Provenance | **57.2 ± 37.8 / 100** | [49.5, 64.9] |
| **LLM-as-Judge** | Overall | **65.7 ± 21.7 / 100** | [61.4, 69.9] |

### Table 2: Performance Breakdown across 5 Categories (20 Questions each)
| Category | Questions | Classification Accuracy | ROUGE-L | Faithfulness | Overall (/100) |
|---|:---:|:---:|:---:|:---:|:---:|
| **DL** | 20 | 30.0\% | 0.085 | 51.0 | **66.2** |
| **ML** | 20 | 20.0\% | 0.080 | 39.0 | **61.8** |
| **MultiSource Conflict** | 20 | 50.0\% | 0.073 | 51.8 | **63.4** |
| **RAG** | 20 | 45.0\% | 0.069 | 45.8 | **65.9** |
| **Transformers** | 20 | 35.0\% | 0.075 | 67.5 | **71.0** |

### Table 3: Performance Breakdown by Difficulty
| Difficulty | Questions | Classification Accuracy | ROUGE-L | Faithfulness | Overall (/100) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Easy** | 24 | 25.0\% | 0.085 | 69.2 | **78.3** |
| **Medium** | 50 | 36.0\% | 0.073 | 43.0 | **62.5** |
| **Hard** | 26 | 46.2\% | 0.075 | 49.6 | **60.1** |

### Table 4: Per-Class Classification (Agree vs Partial vs Conflict)
| Target Class | Support ($N$) | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Agree** | 59 | 0.550 | 0.186 | **0.279** |
| **Partial** | 30 | 0.329 | 0.767 | **0.460** |
| **Conflict** | 11 | 0.200 | 0.182 | **0.191** |