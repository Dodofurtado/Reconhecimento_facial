import cv2
import os
from datetime import datetime

def criar_pasta(nome):
    caminho = os.path.join("dataset_rosto", nome)
    os.makedirs(caminho, exist_ok=True)
    return caminho

def capturar_fotos():
    while True:
        pessoa = input("Digite o nome da pessoa (ou Enter para sair): ").strip()
        if not pessoa:
            break
            
        pasta = criar_pasta(pessoa)
        cap = cv2.VideoCapture(0)
        
        print(f"\nCapturando fotos de: {pessoa}")
        print("Pressione:")
        print("'f' - Tirar foto")
        print("'q' - Próxima pessoa")
        print("'esc' - Sair\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Cria uma cópia para visualização
            frame_vis = frame.copy()
            
            # Desenha guia central apenas na visualização
            h, w = frame_vis.shape[:2]
            centro = (w // 2, h // 2)
            tam = (int(w * 0.25), int(h * 0.45))
            cv2.ellipse(frame_vis, centro, tam, 0, 0, 360, (0, 255, 0), 2)
            
            cv2.imshow('Captura', frame_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                # Salva região do rosto do frame original (sem o arco verde)
                x1 = centro[0] - tam[0]
                y1 = centro[1] - tam[1]
                x2 = centro[0] + tam[0]
                y2 = centro[1] + tam[1]
                rosto = frame[y1:y2, x1:x2]  # Usa frame original, não frame_vis
                
                nome_arquivo = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                caminho = os.path.join(pasta, nome_arquivo)
                cv2.imwrite(caminho, rosto)
                print(f"Foto salva: {nome_arquivo}")
                
            elif key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if key == 27:  # ESC
            break

    print("\nTreinando modelo...")
    os.system('python treinar_rosto.py')
    print("Concluído!")

if __name__ == "__main__":
    capturar_fotos()
