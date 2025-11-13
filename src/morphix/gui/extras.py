import importlib.util

# ensure gui extras are installed
specs = [importlib.util.find_spec(name) for name in ["pygfx", "PyQt5", "rendercanvas"]]
if not all(specs):
    raise ImportError(
        "This module requires additional dependencies to be installed. "
        "Please install 'morphix[gui]'."
    )
