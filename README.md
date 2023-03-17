# NLP Project 2023

Code for *A baseline for detecting Textual Attacks in Sentiment Analysis Classification using Density Estimation*.

* Vincent Nguyen
* Solal Jarreau


## Useful resources
* Course repository : 
  * https://github.com/PierreColombo/NLP_ENSAE_2023
  * https://github.com/PierreColombo/NLP_ENSAE_2023/blob/main/project/project_2_attacks.md
  * https://github.com/PierreColombo/NLP_ENSAE_2023/blob/main/project/project_general_instructions.md
* Main paper : https://arxiv.org/abs/2203.01677
  * https://github.com/anoymous92874838/text-adv-detection
  * https://github.com/bangawayoo/adversarial-examples-in-text-classification
* TextAttack : 
  * https://github.com/QData/TextAttack
  * https://textattack.readthedocs.io/en/latest/
  
 ## Dataset
We work on IMDb loaded from HuggingFace, with train and test split. 

Because creating attacks is costly, attacks are loaded from this repo : https://github.com/bangawayoo/adversarial-examples-in-text-classification. Unzip the file and save it in ```/data```.

## Usage
Procedure is divided in four parts : 
* train_test_split : create training and test samples of imdb and save them in ```/pickle``` folder. 
* embeddings : create embeddings and save them in ```/embeddings``` folder. 
* detection : apply adversarial detection and save results in ```/results``` folder. 
* eval : evaluate the detection method. 

```imdb_bert``` notebooks apply our adversarial detection only for BERT embeddings and a specific attack *TextFooler* while ```global``` notebook are compatible with different transformers architectures and different attacks. 
