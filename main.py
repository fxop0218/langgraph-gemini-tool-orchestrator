from __future__ import annotations
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Tools import
from tools.fake_prods import (
    get_all_products,
    get_product_by_id,
    get_all_users,
    create_user,
    get_user_by_id,
)
from tools.product_list import export_products as _export_sorted_products
from tools.csv_editor import edit_products_csv as _edit_products_csv
from tools.excel_editor import edit_products_excel as _edit_products_excel

# Load environment variables from .env
load_dotenv()


class State(TypedDict):
    """Graph state container.

    Attributes
    ----------
    messages : list
        Accumulated conversation history. The list is managed by `add_messages`
        to ensure proper LangGraph semantics (append-only with tool routing).
    """

    messages: Annotated[list, add_messages]


@tool
def list_products() -> list:
    """Return the full catalog of products.

    Short Description (for LLMs)
    ----------------------------
    Returns the complete product list as a JSON-serializable array.

    Returns
    -------
    list
        List of product dicts. Each item typically contains keys like:
        `id`, `title`, `price`, `category`, `description`, `image`, and `rating`.
    """
    return get_all_products()


@tool
def get_product(product_id: int) -> dict:
    """Retrieve product details by ID.

    Short Description (for LLMs)
    ----------------------------
    Fetch a single product given its numeric `product_id`.

    Parameters
    ----------
    product_id : int
        Unique numeric identifier of the product.

    Returns
    -------
    dict
        Product record if found. May raise if the ID does not exist.

    Raises
    ------
    KeyError
        If the product is not found by the underlying data source.
    """
    return get_product_by_id(product_id)


@tool
def list_users() -> list:
    """Return all registered users.

    Short Description (for LLMs)
    ----------------------------
    Lists all users as JSON-serializable objects.

    Returns
    -------
    list
        List of user dicts with canonical fields such as `id`, `username`,
        `email`, and other profile attributes in the backing store.
    """
    return get_all_users()


@tool
def add_user(user_data: dict) -> dict:
    """Create a new user.

    Short Description (for LLMs)
    ----------------------------
    Persists a user with required fields and returns the created record.

    Parameters
    ----------
    user_data : dict
        Input payload. Required keys:
        - `id` (int)
        - `username` (str)
        - `email` (str)
        - `password` (str)

    Returns
    -------
    dict
        Newly created user payload as stored by the data layer.

    Raises
    ------
    ValueError
        If required keys are missing or invalid.
    """
    return create_user(user_data)


@tool
def get_user(user_id: int) -> dict:
    """Retrieve a user by ID.

    Short Description (for LLMs)
    ----------------------------
    Fetch a single user record by numeric `user_id`.

    Parameters
    ----------
    user_id : int
        Unique numeric identifier of the user.

    Returns
    -------
    dict
        User record if found.

    Raises
    ------
    KeyError
        If the user does not exist.
    """
    return get_user_by_id(user_id)


@tool
def export_sorted_products(
    payload: List[Dict[str, Any]],
    order_by: str,
    file_name: str,
    file_format: str = "xlsx",
) -> dict:
    """Export products to CSV/XLSX with deterministic sorting.

    Short Description (for LLMs)
    ----------------------------
    Sort products by a policy (alphabetical/price/rating/category) and export to a file.

    Parameters
    ----------
    payload : list of dict
        Array of product objects. Missing rating fields are tolerated.
    order_by : {"alphabetical", "price", "rating", "category"}
        Sorting policy (case-insensitive).
    file_name : str
        Safe base name (no paths). Extension is forced by `file_format`.
    file_format : {"csv", "xlsx"}, default "xlsx"
        Output format.

    Returns
    -------
    dict
        {
          "path": str,           # absolute output path
          "rows": int,           # number of exported rows
          "columns": list[str],  # exported columns
          "order_by": str,       # applied sorting criterion
          "file_format": str     # resulting format
        }

    Notes
    -----
    The underlying implementation ensures stable sorting (`mergesort`) and
    safe filename sanitation.
    """
    out_path = _export_sorted_products(payload, order_by, file_name, file_format)
    # Best-effort metadata build; avoid heavy I/O
    # Defer reading back the file to keep the tool side-effect light.
    meta = {
        "path": out_path,
        "rows": len(payload) if isinstance(payload, list) else 0,
        "columns": (
            list(payload[0].keys()) if payload and isinstance(payload[0], dict) else []
        ),
        "order_by": order_by,
        "file_format": file_format.lower(),
    }
    return meta


@tool
def edit_products_file(
    input_csv: str,
    output_name: str,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    dedupe_on: Optional[List[str]] = None,
    sort_order: Optional[str] = None,
    file_format: str = "csv",
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> Dict[str, Any]:
    """Edit and export a product table (CSV→CSV/XLSX) with filters and sorting.

    Short Description (for LLMs)
    ----------------------------
    Loads a CSV, applies filters (category/price/rating), column ops, dedupe, sort, and writes to CSV/XLSX.

    Parameters
    ----------
    input_csv : str
        Existing CSV path to read.
    output_name : str
        Base name only (no paths). Extension derived from `file_format`.
    include_categories, exclude_categories : list[str], optional
        Case-insensitive category filters.
    min_price, max_price : float, optional
        Price range boundaries (inclusive).
    min_rating : float, optional
        Minimum `rating_rate`.
    keep_columns : list[str], optional
        Whitelist columns (intersection with existing).
    drop_columns : list[str], optional
        Columns to drop if present.
    dedupe_on : list[str], optional
        Subset of columns to consider for duplicate removal. If None, full-row dedupe.
    sort_order : {"alphabetical", "price", "rating", "category"}, optional
        Sorting policy.
    file_format : {"csv", "xlsx"}, default "csv"
        Output format.
    encoding : str, default "utf-8-sig"
        CSV encoding (Excel-friendly).
    delimiter : str, default ","
        CSV delimiter.

    Returns
    -------
    dict
        {
          "path": str,            # absolute output path
          "rows_before": int,
          "rows_after": int,
          "columns": list[str],
          "applied": list[str]
        }

    Raises
    ------
    FileNotFoundError
        If `input_csv` does not exist.
    ValueError
        On invalid parameters or unsafe `output_name`.
    """
    return _edit_products_csv(
        input_csv=input_csv,
        output_name=output_name,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        dedupe_on=dedupe_on,
        sort_order=sort_order,
        file_format=file_format,
        encoding=encoding,
        delimiter=delimiter,
    )


@tool
def edit_products_excel(
    input_name: Optional[str] = None,
    input_excel: Optional[str] = None,
    output_name: Optional[str] = None,
    in_place: bool = False,
    sheet_name: Optional[str] = None,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    dedupe_on: Optional[List[str]] = None,
    sort_order: Optional[str] = None,
    output_sheet: str = "Edited",
    remove_ids: Optional[List] = None,
    remove_equals: Optional[Dict[str, List]] = None,
    remove_contains: Optional[Dict[str, List[str]]] = None,
    remove_regex: Optional[Dict[str, str]] = None,
    remove_nulls_in: Optional[List[str]] = None,
    remove_zero_in: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Edit an Excel workbook by name or explicit path and persist changes.

    Short Description (for LLMs)
    ----------------------------
    Reads `.xlsx`, applies filters and deletion rules, column ops, dedupe, sort, and writes back.

    Parameters
    ----------
    input_name : str, optional
        Safe base name under `files/excel/` (preferred; no paths).
    input_excel : str, optional
        Explicit file path (legacy). Ignored if `input_name` is provided.
    output_name : str, optional
        Base name for output (ignored if `in_place=True`).
    in_place : bool, default False
        If True, overwrites the input safely (temp file + atomic replace).
    sheet_name : str, optional
        Sheet to read. If None, default sheet is used.
    include_categories, exclude_categories : list[str], optional
        Category filters (case-insensitive).
    min_price, max_price : float, optional
        Price range boundaries (inclusive).
    min_rating : float, optional
        Minimum `rating_rate`.
    keep_columns, drop_columns : list[str], optional
        Column-level selection/removal.
    dedupe_on : list[str], optional
        Columns used for duplicate removal. If None, full-row dedupe.
    sort_order : {"alphabetical", "price", "rating", "category"}, optional
        Sorting policy.
    output_sheet : str, default "Edited"
        Destination sheet name (max 31 chars).
    remove_ids : list, optional
        Remove rows whose `id` is in the provided list.
    remove_equals : dict[str, list], optional
        Remove rows where `df[col]` is in list for each specified column.
    remove_contains : dict[str, list[str]], optional
        Remove rows if any substring matches (case-insensitive) in specified columns.
    remove_regex : dict[str, str], optional
        Remove rows if regex matches in specified columns.
    remove_nulls_in : list[str], optional
        Drop rows with NULLs in any of the given columns.
    remove_zero_in : list[str], optional
        Drop rows where ANY of the numeric columns equals zero.

    Returns
    -------
    dict
        {
          "path": str,
          "rows_before": int,
          "rows_after": int,
          "columns": list[str],
          "applied": list[str],
          "in_place": bool
        }

    Raises
    ------
    FileNotFoundError
        If the input `.xlsx` cannot be found.
    ValueError
        On invalid parameters or unsafe filenames.
    """
    return _edit_products_excel(
        input_excel=input_excel,
        output_name=output_name,
        input_name=input_name,
        in_place=in_place,
        sheet_name=sheet_name,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        dedupe_on=dedupe_on,
        sort_order=sort_order,
        output_sheet=output_sheet,
        engine_read=None,
        remove_ids=remove_ids,
        remove_equals=remove_equals,
        remove_contains=remove_contains,
        remove_regex=remove_regex,
        remove_nulls_in=remove_nulls_in,
        remove_zero_in=remove_zero_in,
    )


TOOLS = [
    list_products,
    get_product,
    list_users,
    add_user,
    get_user,
    export_sorted_products,
]


def configure_llm():
    """Initialize Gemini 2.5 Flash client.

    Returns
    -------
    langchain.chat_models.base.BaseChatModel
        Configured chat model bound to Google GenAI.

    Raises
    ------
    RuntimeError
        If `GOOGLE_API_KEY` is not present in the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY must be set in the environment (.env).")
    return init_chat_model(model="google_genai:gemini-2.5-flash", api_key=api_key)


def build_agent():
    """Build and compile the LangGraph agent with tool routing.

    Pipeline
    --------
    - Bind tools to the LLM.
    - Add `chatbot` (model inference) and `tools` (execution) nodes.
    - Conditional edge from `chatbot` to `tools` using `tools_condition`.
    - Feedback edge from `tools` back to `chatbot`.
    - In-memory checkpointing via `MemorySaver`.

    Returns
    -------
    langgraph.graph.compiler.CompiledGraph
        Ready-to-invoke graph with checkpointing enabled.
    """
    llm = configure_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def chatbot(state: State):
        """Single-step LLM invocation node compatible with LangGraph state."""
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

    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


def chat():
    """Simple CLI loop to interact with the agent.

    Commands
    --------
    /session <id> : switch ephemeral `thread_id`
    /whoami       : print current `thread_id`
    exit/quit     : terminate loop

    Side Effects
    ------------
    Prints agent outputs to stdout and reads user input from stdin.
    """
    agent = build_agent()
    session_id = "runtime-session"

    print("[Agent initialized: Google Gemini 2.5 Flash + LangGraph]")
    print("Commands: /session <id>  |  /whoami  |  exit\n")
    print(f"(Ephemeral session: {session_id})\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Agent: Session ended. Goodbye.")
            break

        if user_input.startswith("/session "):
            session_id = user_input.split(" ", 1)[1].strip() or session_id
            print(f"Agent: Switched ephemeral session to: {session_id}")
            continue
        if user_input == "/whoami":
            print(f"Agent: current thread_id = {session_id}")
            continue

        state = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": session_id}},
        )

        messages = state.get("messages", [])
        if not messages:
            print("Agent: (no response)")
            continue

        reply = messages[-1]
        if isinstance(reply, ToolMessage):
            print(f"Agent [tool {reply.name}]: {reply.content}")
        else:
            print(f"Agent: {reply.content}")


if __name__ == "__main__":
    chat()
