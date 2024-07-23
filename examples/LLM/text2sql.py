from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import os
from env_variables import MODEL_PATH, BDD_STRUCTURE_PATH

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

    response = llm.invoke(formatted_prompt)
    return response


if __name__ == '__main__':
    # Chemin vers le fichier de schéma
    schema_file_path = BDD_STRUCTURE_PATH
    schema = read_schema(schema_file_path)

    # Exemple de requête en langage naturel
    question = "Find the names and emails of users who ordered products after 2021"

    # Générer la requête SQL
    sql_query = generate_sql_query(question, schema)
    print(sql_query)
