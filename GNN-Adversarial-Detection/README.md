# A-GNN-based-Adversarial-IoT-Malware-Detection-Framework
## Storing CFGs in json files
In order to extract and store the CFG data of a given executable we used the code in `write_graph_json.ipynb` which iw written based on [angr](https://docs.angr.io/). The required packages for angr are available in `angr_requirements.txt` (For extracting the CFGs and then training the GNN models two distinct Anaconda environments were used).  
## Training and testing the classifier and the adversarial detector
For training and evaluating the classifier and the adversarial detector we used [PyG](https://pytorch-geometric.readthedocs.io/). The requried packages are available in `pyg_requirements`. Use `classifier.ipynb` for the classifier and `detector.ipynb` for the detector.