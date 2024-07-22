from langchain_community.llms import LlamaCpp
from env_variables import MODEL_PATH
import gradio as gr
import logging
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s')

# Initialisation du modèle LLM
try:
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.75,
        max_tokens=500,
        top_p=1,
        verbose=True,
    )
    logging.info("Modèle LLM initialisé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de l'initialisation du modèle LLM: {e}")
    raise


def generate_response_stream(question):
    logging.info(f"Question reçue : {question}")
    try:
        response_generator = llm.stream(question)
        for chunk in response_generator:

            logging.info(f"Mot généré : {chunk}")
            yield chunk
    except Exception as e:
        error_message = f"Une erreur s'est produite lors de la génération de la réponse : {e}"
        logging.error(error_message)
        yield error_message


def start_generation(question):
    response = ""
    for word in generate_response_stream(question):
        response += word
        yield response # On retire les espaces en trop


# Interface Gradio
with gr.Blocks() as iface:
    with gr.Row():
        question_input = gr.Textbox(label="Posez votre question ici")
        start_button = gr.Button("Commencer la génération")

    response_display = gr.Textbox(label="Réponse", interactive=False)

    start_button.click(fn=start_generation, inputs=question_input, outputs=response_display)

if __name__ == "__main__":
    logging.info("Démarrage de l'interface Gradio...")
    iface.launch()
