import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_agent


def main():
    # ---------------------------------------------------------
    # 1. Comprobar token de Hugging Face
    # ---------------------------------------------------------
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        print("Error: no se encontró HUGGINGFACEHUB_API_TOKEN.")
        print("Ejecuta:")
        print("export HUGGINGFACEHUB_API_TOKEN='tu_token'")
        return

    print("Iniciando el agente investigador...\n")

    # ---------------------------------------------------------
    # 2. Modelo de Hugging Face
    # ---------------------------------------------------------
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        provider="auto",
        temperature=0.1,
        max_new_tokens=512,
    )

    llm = ChatHuggingFace(
        llm=llm_endpoint
    )

    # ---------------------------------------------------------
    # 3. Herramienta de Wikipedia
    # ---------------------------------------------------------
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=3000,
    )

    herramienta_wiki = WikipediaQueryRun(
        api_wrapper=api_wrapper
    )

    herramientas = [
        herramienta_wiki
    ]

    # ---------------------------------------------------------
    # 4. Crear agente
    # ---------------------------------------------------------
    agente = create_agent(
        model=llm,
        tools=herramientas,
        system_prompt=(
            "Eres un agente investigador. "
            "Responde en español de forma clara y precisa. "
            "Cuando necesites información sobre una persona, "
            "lugar, concepto o acontecimiento, utiliza Wikipedia. "
            "No inventes información. "
            "Si la herramienta no proporciona suficiente información, "
            "indica claramente que no puedes confirmarla."
        ),
    )

    print("Agente listo. Escribe 'salir' para terminar.\n")

    # ---------------------------------------------------------
    # 5. Bucle de conversación
    # ---------------------------------------------------------
    while True:
        pregunta = input("\nPregunta para el agente: ").strip()

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Saliendo del programa...")
            break

        if not pregunta:
            continue

        try:
            resultado = agente.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": pregunta,
                        }
                    ]
                }
            )

            # El último mensaje contiene la respuesta final
            respuesta_final = resultado["messages"][-1].content

            print("\n" + "=" * 60)
            print("RESPUESTA FINAL:")
            print("=" * 60)
            print(respuesta_final)
            print("=" * 60)

        except Exception as e:
            print("\nOcurrió un error al ejecutar el agente:")
            print(e)


if __name__ == "__main__":
    main()