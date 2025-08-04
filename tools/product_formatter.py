def list_products(json_products):
    """
    Enumerate and format a list of products from JSON data.

    Args:
        json_products (list of dict): A list of product objects, each containing at least:
            - title (str): Name of the product.
            - price (int|float): Price of the product.
            - category (str): Category of the product.

    Returns:
        list of str: A numbered list of formatted product descriptions.
    """
    formatted_list = []
    for index, product in enumerate(json_products, start=1):
        title = product.get("title", "N/A")
        price = product.get("price", "N/A")
        category = product.get("category", "N/A")
        formatted_list.append(f"{index}. {title} — ${price} ({category})")
    return formatted_list


if __name__ == "__main__":
    import json
    import sys

    # Load JSON from a file or stdin
    data = json.load(sys.stdin) if not sys.argv[1:] else json.load(open(sys.argv[1]))

    # Generate and print formatted list
    for line in list_products(data):
        print(line)
