import cv2
import joblib
import numpy as np

def reconhecer_rosto():
    # Carrega o modelo treinado
    print("Carregando modelo...")
    try:
        modelo = joblib.load("modelo_rosto.joblib")
    except:
        print("Erro: Modelo não encontrado! Execute primeiro o treinar_rosto.py")
        return

    # Inicia a webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam!")
        return

    print("\nPressione:")
    print("'q' - Reconhecer rosto")
    print("'esc' - Sair\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Desenha guia central
        h, w = frame.shape[:2]
        centro = (w // 2, h // 2)
        tam = (int(w * 0.25), int(h * 0.45))
        cv2.ellipse(frame, centro, tam, 0, 0, 360, (0, 255, 0), 2)
        
        cv2.imshow('Reconhecimento', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Captura região do rosto
            x1 = centro[0] - tam[0]
            y1 = centro[1] - tam[1]
            x2 = centro[0] + tam[0]
            y2 = centro[1] + tam[1]
            rosto = frame[y1:y2, x1:x2]
            
            # Prepara a imagem
            rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
            rosto = cv2.resize(rosto, (64, 64)).flatten().reshape(1, -1)
            
            # Faz a predição
            probabilidades = modelo.predict_proba(rosto)[0]
            idx_max = np.argmax(probabilidades)
            confianca = probabilidades[idx_max]
            nome = modelo.classes_[idx_max]
            
            # Mostra resultado
            if confianca > 0.6:
                print(f"\nPessoa reconhecida: {nome}")
                print(f"Confiança: {confianca:.2%}")
            else:
                print("\nPessoa não reconhecida")
                print(f"Confiança muito baixa: {confianca:.2%}")
            
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconhecer_rosto()
