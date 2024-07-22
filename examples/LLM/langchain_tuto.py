from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import os
from env_variables import MODEL_PATH

if __name__ == '__main__':
    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate.from_template(template)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    #other mistral model https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF
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

    question = """
    Question: A rap battle between Stephen Colbert and John Oliver
    """
    llm.invoke(question)
