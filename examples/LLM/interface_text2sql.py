from langchain_community.llms import LlamaCpp
from env_variables import MODEL_PATH
import gradio as gr
import logging
import time
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import os
from env_variables import MODEL_PATH, BDD_STRUCTURE_PATH

# Configuration du logging
logging.basicConfig(level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s')

# Lire le schéma de la base de données depuis un fichier texte
def read_schema(file_path):
    with open(file_path, 'r') as file:
        schema = file.read()
    return schema


# Fonction pour générer la requête SQL
def generate_sql_query(question, schema):
    template = """Generate an SQL query for the following natural language request: '{question}'
    Database schema:
    {schema}
    SQL query:"""

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(question=question, schema=schema)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    print(MODEL_PATH)
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    response_generator = llm.stream(formatted_prompt)
    for chunk in response_generator:

            logging.info(f"Mot généré : {chunk}")
            yield chunk

def start_generation(question):
    schema = read_schema(BDD_STRUCTURE_PATH)
    response = ""
    for word in generate_sql_query(question, schema):
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
