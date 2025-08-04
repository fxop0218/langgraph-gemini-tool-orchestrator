import requests

# Base URL for FakeStore API
BASE_URL = "https://fakestoreapi.com"

# Product Endpoints


def get_all_products():
    """
    Retrieve all products from the FakeStore API.
    Returns:
        list: A list of product objects (JSON).
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/products")
    response.raise_for_status()
    return response.json()


def get_product_by_id(product_id):
    """
    Retrieve a single product by its ID.
    Args:
        product_id (int): The unique identifier of the product.
    Returns:
        dict: The product object corresponding to the provided ID.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/products/{product_id}")
    response.raise_for_status()
    return response.json()


def create_product(product_data):
    """
    Create a new product in the FakeStore.
    Args:
        product_data (dict): A dictionary containing product fields:
            - id (int)
            - title (str)
            - price (float)
            - description (str)
            - category (str)
            - image (str): URL to the product image.
    Returns:
        dict: The created product object as returned by the API.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.post(f"{BASE_URL}/products", json=product_data)
    response.raise_for_status()
    return response.json()


def update_product(product_id, update_data):
    """
    Update an existing product's details.
    Args:
        product_id (int): The unique identifier of the product to update.
        update_data (dict): A dictionary with updated product fields.
    Returns:
        dict: The updated product object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.put(f"{BASE_URL}/products/{product_id}", json=update_data)
    response.raise_for_status()
    return response.json()


def delete_product(product_id):
    """
    Delete a product from the FakeStore.
    Args:
        product_id (int): The unique identifier of the product to remove.
    Returns:
        dict: Confirmation of deletion or deleted object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.delete(f"{BASE_URL}/products/{product_id}")
    response.raise_for_status()
    return response.json()


# Cart Endpoints


def get_all_carts():
    """
    Retrieve all shopping carts from the FakeStore API.
    Returns:
        list: A list of cart objects.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/carts")
    response.raise_for_status()
    return response.json()


def get_cart_by_id(cart_id):
    """
    Retrieve a specific cart by its ID.
    Args:
        cart_id (int): The unique identifier of the cart.
    Returns:
        dict: The cart object corresponding to the provided ID.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/carts/{cart_id}")
    response.raise_for_status()
    return response.json()


def create_cart(cart_data):
    """
    Create a new shopping cart.
    Args:
        cart_data (dict): A dictionary containing cart fields:
            - id (int)
            - userId (int)
            - products (list of dict): List of products with their quantities.
    Returns:
        dict: The created cart object as returned by the API.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.post(f"{BASE_URL}/carts", json=cart_data)
    response.raise_for_status()
    return response.json()


def update_cart(cart_id, update_data):
    """
    Modify an existing shopping cart.
    Args:
        cart_id (int): The unique identifier of the cart to update.
        update_data (dict): A dictionary with updated cart fields.
    Returns:
        dict: The updated cart object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.put(f"{BASE_URL}/carts/{cart_id}", json=update_data)
    response.raise_for_status()
    return response.json()


def delete_cart(cart_id):
    """
    Remove a shopping cart from the FakeStore.
    Args:
        cart_id (int): The unique identifier of the cart to remove.
    Returns:
        dict: Confirmation of deletion or deleted object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.delete(f"{BASE_URL}/carts/{cart_id}")
    response.raise_for_status()
    return response.json()


# User Endpoints


def get_all_users():
    """
    Retrieve all registered users from the FakeStore API.
    Returns:
        list: A list of user objects.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/users")
    response.raise_for_status()
    return response.json()


def get_user_by_id(user_id):
    """
    Retrieve a single user by their ID.
    Args:
        user_id (int): The unique identifier of the user.
    Returns:
        dict: The user object corresponding to the provided ID.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(f"{BASE_URL}/users/{user_id}")
    response.raise_for_status()
    return response.json()


def create_user(user_data):
    """
    Register a new user in the FakeStore.
    Args:
        user_data (dict): A dictionary containing user fields:
            - id (int)
            - username (str)
            - email (str)
            - password (str)
    Returns:
        dict: The created user object as returned by the API.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.post(f"{BASE_URL}/users", json=user_data)
    response.raise_for_status()
    return response.json()


def update_user(user_id, update_data):
    """
    Update details for an existing user.
    Args:
        user_id (int): The unique identifier of the user to update.
        update_data (dict): A dictionary with updated user fields.
    Returns:
        dict: The updated user object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.put(f"{BASE_URL}/users/{user_id}", json=update_data)
    response.raise_for_status()
    return response.json()


def delete_user(user_id):
    """
    Delete a user from the FakeStore system.
    Args:
        user_id (int): The unique identifier of the user to delete.
    Returns:
        dict: Confirmation of deletion or deleted object.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    response = requests.delete(f"{BASE_URL}/users/{user_id}")
    response.raise_for_status()
    return response.json()


# Authentication Endpoint


def login(username, password):
    """
    Authenticate a user and retrieve an access token.
    Args:
        username (str): The user's username.
        password (str): The user's password.
    Returns:
        dict: Authentication token object with "token" key.
    Raises:
        HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    credentials = {"username": username, "password": password}
    response = requests.post(f"{BASE_URL}/auth/login", json=credentials)
    response.raise_for_status()
    return response.json()
