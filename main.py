"""
Agente basado en LangGraph con Google Gemini y herramientas para productos y usuarios.
Carga variables de entorno usando python-dotenv.
Incluye flujos específicos para creación, listado y búsqueda de usuarios por ID sin depender únicamente del LLM.
"""

from dotenv import load_dotenv
import os
import re
import json
import pathlib
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import List, Dict, Any
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


from tools.product_list import export_products as _export_sorted_products

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


@tool
def export_sorted_products(
    payload: List[Dict[str, Any]],
    order_by: str,
    file_name: str,
    file_format: str = "xlsx",
) -> dict:
    """
    Export a product list to CSV/XLSX with a chosen sorting policy.

    Args:
        payload: List of product dicts (as provided in your example).
        order_by: "alphabetical" | "price" | "rating" | "category".
        file_name: Base file name only (no paths). Extension forced by file_format.
        file_format: "csv" or "xlsx" (default "xlsx").

    Returns:
        dict with:
          - path: absolute path of the created file
          - rows: number of exported rows
          - columns: exported columns
          - order_by: applied criterion
          - file_format: final format
    """
    print(payload)
    return _export_sorted_products(payload, order_by, file_name, file_format)


# Lista de todas las herramientas disponibles
TOOLS = [
    list_products,
    get_product,
    list_users,
    add_user,
    get_user,
    export_sorted_products,
]

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

    # ✅ memoria en RAM (no persiste tras cerrar el proceso)
    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


# --- Bucle de interacción -----------------------------------------------------


def chat():
    agent = build_agent()
    # un id cualquiera por ejecución; si quieres, cámbialo por timestamp/uuid
    session_id = "runtime-session"

    pending_user_creation = False
    print("[Agente iniciado con Google Gemini 2.5 Flash y LangGraph]")
    print("Comandos: /session <id>  |  /whoami  |  exit\n")
    print(f"(Sesión actual efímera: {session_id})\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Agent: Sesión finalizada. Hasta pronto.")
            break

        # cambiar de hilo efímero dentro de la misma instancia (opcional)
        if user_input.startswith("/session "):
            session_id = user_input.split(" ", 1)[1].strip() or session_id
            print(f"Agent: Cambiada la sesión efímera a: {session_id}")
            continue
        if user_input == "/whoami":
            print(f"Agent: thread_id actual = {session_id}")
            continue

        # ... (tus flujos manuales: crear/listar/buscar usuarios) ...

        # ✅ IMPORTANTE: pasar thread_id en config.configurable
        state = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": session_id}},
        )

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
