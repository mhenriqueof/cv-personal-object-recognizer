def input_object_name() -> str:
    """
    Gets a valid object name from user via console input.
    
    Returns:
        Non-empty string with max 30 characters.
    """
    while True:
        object_name = input("Object name: ").strip()
        if len(object_name) == 0:
            print("Object name cannot be empty.")
        elif len(object_name) > 30:
            print("Name too long. (max 30 characters)")
        else:
            return object_name
        