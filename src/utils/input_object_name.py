def input_object_name() -> str:
    while True:
        object_name = input("Object name: ").strip()
        if len(object_name) != 0:
            return object_name
        print("Type a valid name.")
        