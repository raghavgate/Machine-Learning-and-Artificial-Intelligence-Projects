# Machine Learning and Artificial Intelligence Projects

Notebooks from coursework and independent projects, from neural-network fundamentals to retrieval-augmented generation and agent workflows. Each notebook opens in Google Colab from the badge at its top.

## Index

| Notebook | What it does | Stack | Result recorded in the notebook |
|---|---|---|---|
| [GraphRAG_EECS295](GraphRAG_EECS295.ipynb) | Graph retrieval-augmented generation for multi-hop QA. Extracts entities with spaCy into a NetworkX knowledge graph, retrieves k-hop subgraphs for query entities, and compares answers against a chunked FAISS vector-retrieval baseline. Scores both with exact match and token F1. | transformers (FLAN-T5 or Mistral-7B-Instruct, 4-bit), sentence-transformers, FAISS, spaCy, NetworkX | Per-question F1 for vanilla RAG vs. GraphRAG on sample multi-hop questions |
| [Agentic_Workflow_for_News_Analysis](Agentic_Workflow_for_News_Analysis.ipynb) | A staged agent that ingests news, summarizes it, analyzes sentiment, evaluates risk, decides on an action, and acts, passing a typed state between steps. | transformers, typed state | Qualitative run output |
| [Mixed_Reality_Context_Aware_Modeling_Project](Mixed_Reality_Context_Aware_Modeling_Project.ipynb) | Unsupervised analysis of collaboration sessions. Clusters participants on network-centrality features to assign leader-like, follower-like and neutral roles, maps role-to-role interactions as a heatmap, and relates each group's role mix to task completion time. | pandas, scikit-learn, seaborn, matplotlib | Role clusters, interaction heatmap, role mix vs. completion time across 12 groups |
| [Fashion_Image_Classification…Fashion_MNIST](Fashion_Image_Classification_Using_Deep_Artificial_and_Convolutional_Neural_Networks_on_Fashion_MNIST.ipynb) | Dense and convolutional networks for 10-class Fashion-MNIST, with data augmentation, a validation split, and a confusion matrix. | TensorFlow/Keras, scikit-learn, seaborn | Test accuracy 0.864 |
| [Neural-Networks/Convolutional Neural Network for Clothing Image Recognition](Neural-Networks/Convolutional%20Neural%20Network%20for%20Clothing%20Image%20Recognition.ipynb) | A later run of the Fashion-MNIST CNN. | TensorFlow/Keras | Test accuracy 0.866 |
| [Neural-Networks/Neural_Net_for_Handwritten_Digit_Recognition](Neural-Networks/Neural_Net_for_Handwritten_Digit_Recognition.ipynb) | Multiclass (0–9) digit recognition with ReLU and softmax, plus a decision-tree exercise (course assignment). | TensorFlow/Keras, NumPy | Training loss curve |
| [Neural-Networks/Neural_Network_for_Binary_Digit_Recognition](Neural-Networks/Neural_Network_for_Binary_Digit_Recognition.ipynb) | Binary (0 vs. 1) digit recognition, with the forward pass implemented both in Keras and in NumPy (course practice lab). | TensorFlow/Keras, NumPy | Training loss curve |

## Running

Open a notebook in Colab with its badge. GraphRAG expects a T4 GPU runtime when `MODEL_CHOICE = "mistral"`; FLAN-T5 also runs on CPU. The two digit-recognition labs import course helper modules (`autils`, `public_tests`, `lab_utils_softmax`) that are not in this repository, so they are kept for reading rather than re-running.

## Notes

- The results in the table are the outputs saved in each notebook, from single runs; they are not benchmarks.
- The two Fashion-MNIST notebooks share most of their code. The second is a re-run with slightly different results.
