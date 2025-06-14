import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def treinar_modelo():
    print("Carregando imagens...")
    X = []  # imagens
    y = []  # labels (nomes)
    
    for pessoa in os.listdir("dataset_rosto"):
        if pessoa.lower() == "desconhecido":
            continue  # Pula a pasta desconhecido
            
        pasta = os.path.join("dataset_rosto", pessoa)
        if not os.path.isdir(pasta):
            continue
            
        print(f"Processando fotos de: {pessoa}")
        for foto in os.listdir(pasta):
            caminho = os.path.join(pasta, foto)
            img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Redimensiona para tamanho padrão
            img = cv2.resize(img, (64, 64)).flatten()
            X.append(img)
            y.append(pessoa)

    if not X:
        print("Nenhuma foto encontrada para treinar!")
        return

    X = np.array(X)
    y = np.array(y)

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo
    print("\nTreinando modelo...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Avalia e salva
    score = clf.score(X_test, y_test)
    print(f"\nAcurácia do modelo: {score:.2%}")
    
    joblib.dump(clf, "modelo_rosto.joblib")
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    treinar_modelo()
