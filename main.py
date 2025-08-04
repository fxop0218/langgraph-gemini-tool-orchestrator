"""
Agente basado en LangGraph con Google Gemini y herramientas para productos y usuarios.
Carga variables de entorno usando python-dotenv.
Incluye flujos específicos para creación, listado y búsqueda de usuarios por ID sin depender únicamente del LLM.
"""

from dotenv import load_dotenv
import os
import re
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tools.fake_prods import (
    get_all_products,
    get_product_by_id,
    get_all_users,
    create_user,
    get_user_by_id,
)

# Carga variables de entorno desde .env
load_dotenv()

# --- Definición del estado del grafo -----------------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- Definición de herramientas con @tool ------------------------------------


@tool
def list_products() -> list:
    """Recupera la lista completa de productos."""
    return get_all_products()


@tool
def get_product(product_id: int) -> dict:
    """Recupera detalle de un producto por su ID."""
    return get_product_by_id(product_id)


@tool
def list_users() -> list:
    """Recupera todos los usuarios registrados."""
    return get_all_users()


@tool
def add_user(user_data: dict) -> dict:
    """
    Crea un nuevo usuario.
    `user_data` debe incluir: id (int), username (str), email (str), password (str).
    """
    return create_user(user_data)


@tool
def get_user(user_id: int) -> dict:
    """Recupera un usuario por su ID."""
    return get_user_by_id(user_id)


# Lista de todas las herramientas disponibles
TOOLS = [list_products, get_product, list_users, add_user, get_user]

# --- Configuración del LLM -------------------------------------


def configure_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Debe definir GOOGLE_API_KEY en el entorno (.env)")
    # Inicializa Gemini 2.5 Flash vía Google Studio
    return init_chat_model(model="google_genai:gemini-2.5-flash", api_key=api_key)


# --- Montaje del grafo de ejecución -------------------------------------------


def build_agent():
    llm = configure_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(TOOLS))

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()


# --- Bucle de interacción -----------------------------------------------------


def chat():
    agent = build_agent()
    pending_user_creation = False

    print("[Agente iniciado con Google Gemini 2.5 Flash y LangGraph]")
    print("Escribe 'exit' o 'quit' para finalizar.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Agent: Sesión finalizada. Hasta pronto.")
            break

        # Flujo manual: iniciar creación de usuario
        if not pending_user_creation and re.search(
            r"crear.*usuario", user_input, re.IGNORECASE
        ):
            pending_user_creation = True
            print(
                "Agent: Para poder crear el nuevo usuario, necesito la siguiente información:"
            )
            print("* ID (número)")
            print("* Nombre de usuario (texto)")
            print("* Correo electrónico (texto)")
            print("* Contraseña (texto)")
            print("Por favor, proporciónamela en una sola línea, separada por comas.\n")
            continue

        # Procesar datos de creación
        if pending_user_creation:
            match = re.search(
                r"id\s*(\d+).*?nombre(?: de usuario)?\s*([^,]+).*?correo(?: electrónico)?\s*([^,]+).*?contraseña\s*(\S+)",
                user_input,
                re.IGNORECASE,
            )
            if match:
                user_data = {
                    "id": int(match.group(1)),
                    "username": match.group(2).strip(),
                    "email": match.group(3).strip(),
                    "password": match.group(4).strip(),
                }
                try:
                    result = add_user(user_data)
                    print("Agent: Usuario creado correctamente:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"Agent: Error al crear el usuario: {e}")
            else:
                print(
                    "Agent: No he podido interpretar los datos. Asegúrate de usar el formato correcto."
                )
            pending_user_creation = False
            continue

        # Flujo manual: listar usuarios
        if re.search(r"listar.*usuarios|ver.*usuarios", user_input, re.IGNORECASE):
            try:
                users = list_users()
                print("Agent: Lista de usuarios registrados:")
                print(json.dumps(users, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Agent: Error al recuperar usuarios: {e}")
            continue

        # Flujo manual: buscar usuario por ID
        match_id = re.search(r"buscar.*usuario.*id\s*(\d+)", user_input, re.IGNORECASE)
        if match_id:
            uid = int(match_id.group(1))
            try:
                user = get_user(uid)
                print(f"Agent: Usuario encontrado (ID {uid}):")
                print(json.dumps(user, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Agent: Error al buscar el usuario: {e}")
            continue

        # Flujo por defecto: paso al agente en grafo
        state = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = state.get("messages", [])
        if not messages:
            print("Agent: (sin respuesta)")
            continue

        reply = messages[-1]
        if isinstance(reply, ToolMessage):
            print(f"Agent [tool {reply.name}]: {reply.content}")
        else:
            print(f"Agent: {reply.content}")


if __name__ == "__main__":
    chat()
