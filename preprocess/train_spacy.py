import spacy
from spacy.training import Example
import random
import os
import pandas as pd

# Configuração de caminhos
DATASET_PATH = "C:/Users/rober/OneDrive/Estudo/Rede Cidadã/Projetos/IA/chatbot-spacy/data/dataset.csv"
MODEL_OUTPUT_PATH = "C:/Users/rober/OneDrive/Estudo/Rede Cidadã/Projetos/IA/chatbot-spacy/models/chatbot_spacy_model"


# Função para carregar e preparar os dados
def load_data(path):
    import pandas as pd
    
    # Carregar o dataset
    data = pd.read_csv(path)
    # dropna() -> Remove arquivos nulos e drop_duplicates() -> Remove duplicados
    data = data[['Pergunta', 'Intenção']].dropna().drop_duplicates()

    # Identificar todas as intenções únicas
    unique_labels = data['Intenção'].unique()

    # Formatar os dados para spaCy
    training_data = []
    for _, row in data.iterrows():
        # Criar categorias binárias (1 para a intenção correta, 0 para as outras)
        # row contém os dados da linha atual
        categories = {label: (1 if label == row['Intenção'] else 0) for label in unique_labels}
        # Adicionando à lista training_data um par (tupla) 
        training_data.append((row['Pergunta'], {"cats": categories}))
    
    # Dados preparados para o treinamento
    # Lista de intenções únicas
    return training_data, unique_labels


# Função para treinar o modelo
# training_data -> dados formatados
# labels -> intenções únicas
# output_path -> caminho onde será salvo o modelo
def train_spacy(training_data, labels, output_path):
    # Criar modelo spacy vazio em português
    nlp = spacy.blank("pt")

    # Adicionar componente de classificação de texto
    # pipeline -> processo o texto para tarefas diferentes, como análise gramatical, reconhecimento de entidades...
    # Adiciona o componente textcat (classificação de texto) ao pipeline do modelo, caso ainda não esteja presente.
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    
    # Adicionar rótulos
    for label in labels:
        textcat.add_label(label)

    # Iniciar treinamento
    nlp.begin_training()

    for epoch in range(1000):  # Número de épocas
        random.shuffle(training_data)
        losses = {}

        # Processar minibatches
        for batch in spacy.util.minibatch(training_data, size=30):
            # Inicializa uma lista para armezenar os exemplos do lote atual
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)  # Criar documento spaCy
                example = Example.from_dict(doc, annotations)  # Criar exemplo
                examples.append(example)
            
            # Atualizar o modelo com o batch de exemplos
            nlp.update(examples, losses=losses)

        print(f"Epoch {epoch} - Losses: {losses}")

    # Salvar o modelo treinado
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    nlp.to_disk(output_path)
    print(f"Modelo salvo em {output_path}")


if __name__ == "__main__":
    # Carregar dados e treinar o modelo
    training_data, labels = load_data(DATASET_PATH)
    train_spacy(training_data, labels, MODEL_OUTPUT_PATH)
