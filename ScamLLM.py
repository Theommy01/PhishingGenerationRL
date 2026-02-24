#https://huggingface.co/phishbot/ScamLLM

"""
Roy Sayak Saha; Thota Poojitha; Naragam Krishna Vamsi; Nilizadeh Shirin

Overview
Our model, "ScamLLM" is designed to identify malicious prompts that can be used to generate phishing websites using popular commercial LLMs like ChatGPT, Bard and Claude. This model is obtained by finetuning a Pre-Trained RoBERTa using a dataset encompassing multiple sets of malicious prompts.

Try out "ScamLLM" using the Inference API. Our model classifies prompts with "Label 1" to signify the identification of a phishing attempt, while "Label 0" denotes a prompt that is considered safe and non-malicious.

Dataset Details
The dataset utilized for training this model has been created using malicious prompts generated using GPT 3.5T and GPT-4. Due to being active vulnerabilities under review, our dataset of malicious prompts is available only upon request at this stage.

Training Details
The model was trained using RobertaForSequenceClassification.from_pretrained. In this process, both the model and tokenizer pertinent to the RoBERTa-base were employed and trained for 10 epochs (learning rate 2e-5 and AdamW Optimizer).
"""

"""
Our model classifies prompts with "Label 1" to signify the identification of a phishing attempt, while "Label 0" denotes a prompt that is considered safe and non-malicious.
"""

from transformers import pipeline

def normalize_outputs(model_outputs):
    # se è annidato (lista contenente la lista dei label), estrai la prima lista
    if len(model_outputs) == 1 and isinstance(model_outputs[0], list):
        labels_list = model_outputs[0]
    else:
        labels_list = model_outputs

    # costruisci un dizionario label -> score
    scores = {item['label']: item['score'] for item in labels_list}
    return scores

print("Working 1")
classifier = pipeline(task="text-classification", model="phishbot/ScamLLM", top_k=None)
print("Working 2")
#prompt = ["Your Sample Sentence or Prompt...."]
prompt = ["Hello, To complete the creation of your account to access the application form, please click on the following activation link: Create my account Username: OMARNAJA36@GMAIL.COM With kind regards, EPFL Registrar's Office Lausanne, Switzerland. For more information, please have a look at https://www.epfl.ch/education/studies/en/support-and-health/student_desk/"]
print("Working 3")
model_outputs = classifier(prompt)
print("Working 4")
i = 0
print(model_outputs[0])

scores = normalize_outputs(model_outputs)

print("prompt ", i, " = ' ", prompt[i], " '")

# Estraiamo il punteggio di "LABEL_1" (malicious) e calcoliamo la percentuale
malicious_score = scores.get('LABEL_1', 0.0)
malicious_percentage = malicious_score * 100

# Formattiamo a 2 cifre decimali e sostituiamo il punto con la virgola
formatted_percentage = f"{malicious_percentage:.2f}".replace('.', ',')

# Ora confronti in sicurezza usando le chiavi
if scores.get('LABEL_0', 0.0) > scores.get('LABEL_1', 0.0):
    print("Prompt", i, "Classified as: safe")
    print(f"Malicious percentage = {formatted_percentage}%\n")
else:
    print("Prompt", i, "Classified as: malicious")
    print(f"Malicious percentage = {formatted_percentage}%\n")

"""
If you use our model in your research, please cite our paper "From Chatbots to Phishbots?: Phishing Scam Generation in Commercial Large Language Models" (https://www.computer.org/csdl/proceedings-article/sp/2024/313000a221/1WPcYLpYFHy).

BibTeX below:

  title={From Chatbots to Phishbots?: Phishing Scam Generation in Commercial Large Language Models},
  author={Roy, Sayak Saha and Thota, Poojitha and Naragam, Krishna Vamsi and Nilizadeh, Shirin},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={221--221},
  year={2024},
  organization={IEEE Computer Society}
}
"""



"""

Cos’è RoBERTa

RoBERTa (A Robustly Optimized BERT Pretraining Approach) è una variante migliorata del modello BERT sviluppata da Facebook AI (Liu et al., 2019).
È un trasformatore bidirezionale pre-addestrato su enormi quantità di testo con l’obiettivo di comprendere il significato contestuale delle parole in una frase.
Rispetto a BERT, RoBERTa elimina alcune restrizioni del training originario e introduce ottimizzazioni che ne aumentano la robustezza e la capacità di generalizzazione.

Dal punto di vista architetturale, RoBERTa mantiene la struttura encoder-only del transformer, basata su più livelli di self-attention, ma differisce da BERT per:

l’uso di dataset di addestramento più ampi (160 GB vs 16 GB di BERT);

addestramento più lungo, con batch di grandi dimensioni;

rimozione dell’obiettivo NSP (Next Sentence Prediction), ritenuto inutile;

dinamic masking, cioè una mascheratura casuale dei token a ogni epoca, che aumenta la varietà del training.

Queste modifiche rendono RoBERTa più capace di comprendere strutture sintattiche e semantiche complesse, quindi più adatto a compiti di text classification e phishing detection basata sul linguaggio naturale.

Vantaggi di RoBERTa

-Eccellente comprensione contestuale del linguaggio

-Grazie alla bidirezionalità e all’addestramento su grandi corpus, RoBERTa coglie relazioni semantiche sottili, rendendolo superiore a modelli classici (SVM, Naive Bayes) che si basano su feature statistiche (TF-IDF, n-gram).

-Alte prestazioni nelle task di classificazione testuale

In diversi benchmark (GLUE, SST-2, MRPC, ecc.), RoBERTa supera costantemente modelli tradizionali e anche BERT, con una migliore capacità di distinguere testi malevoli o ingannevoli da testi benigni.

Adattabilità al dominio della sicurezza

-Può essere facilmente fine-tuned su dataset specifici, come collezioni di phishing e-mail, siti fraudolenti, o prompt malevoli, mantenendo ottime prestazioni anche con dataset di dimensioni moderate.

-Resilienza ai tentativi di evasione linguistica

-Modelli transformer come RoBERTa sono più difficili da ingannare rispetto ai modelli classici, poiché valutano il contesto globale e non solo parole chiave o pattern superficiali.

-Compatibilità con framework open-source

-È disponibile su piattaforme come Hugging Face, integrabile con pochi comandi in PyTorch o TensorFlow, il che facilita la riproducibilità e la sperimentazione.

Svantaggi di RoBERTa

-Elevata complessità computazionale

-Richiede GPU e risorse notevoli per l’addestramento o il fine-tuning; modelli come SVM o Naive Bayes sono molto più leggeri e interpretabili.

-Minor trasparenza del processo decisionale

-Essendo un modello “deep”, RoBERTa è una black box: è difficile comprendere esattamente quali parole o pattern portino alla classificazione “phishing” → richiede tecniche di interpretabilità (es. LIME o SHAP).

Sensibilità al dataset di training

Se il dataset è limitato o sbilanciato, il modello può sovra-adattarsi e diventare meno generalizzabile a nuovi tipi di attacco.

Potenziale vulnerabilità ad attacchi di avvelenamento dei dati o prompt injection

Un avversario può introdurre dati falsi o prompt camuffati per indurre errori di classificazione.

"""
