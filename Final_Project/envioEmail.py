import win32com.client
import os

class AlertaMLOutlook:
    def __init__(self):
        """
        Inicializa a conexão com o Outlook da máquina.
        """
        try:
            self.outlook = win32com.client.Dispatch("Outlook.Application")
            print("✅ Outlook conectado com sucesso.")
        except Exception as e:
            self.outlook = None
            print(f"❌ Erro ao conectar com o Outlook: {e}")

    def enviar_alerta_ml(self, graficos_dir="results/plots"):
        """
        Envia um e-mail informando que a análise de ML foi realizada com anexos dos gráficos.
        """
        if not self.outlook:
            print("⚠️ Outlook não configurado corretamente. O e-mail não será enviado.")
            return

        destinatario = "carineestevao06@gmail.com"
        assunto = "Análise de ML Concluída!"
        mensagem = "Olá, a análise de Machine Learning foi finalizada. Veja os gráficos em anexo."

        anexos = [
            os.path.join(graficos_dir, "roc_curve.png"),
            os.path.join(graficos_dir, "confusion_matrix.png"),
            os.path.join(graficos_dir, "precision_recall_curve.png")
        ]

        try:
            email = self.outlook.CreateItem(0)  
            email.To = destinatario
            email.Subject = assunto
            email.Body = mensagem

            # Adicionar anexos, se existirem
            for arquivo in anexos:
                caminho_absoluto = os.path.abspath(arquivo)
                if os.path.exists(caminho_absoluto):
                    email.Attachments.Add(caminho_absoluto)
                else:
                    print(f"⚠️ Arquivo não encontrado: {caminho_absoluto}")

            # Enviar o e-mail automaticamente
            email.Send()
            print("✅ E-mail enviado com sucesso via Outlook!")

        except Exception as e:
            print(f"❌ Erro ao enviar o e-mail pelo Outlook: {e}")
