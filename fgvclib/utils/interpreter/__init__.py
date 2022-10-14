from .cam import cam


__all__ = ["cam"]

def get_interpreter(interpreter_name):
    r"""Return the interpreter with the given name.

        Args: 
            interpreter_name (str): 
                The name of interpreter.
        
        Return: 
            The interpreter contructor method.
    """
    if interpreter_name not in globals():
        raise NotImplementedError(f"Interpreter not found: {interpreter_name}\nAvailable interpreters: {__all__}")
    return globals()[interpreter_name]
    