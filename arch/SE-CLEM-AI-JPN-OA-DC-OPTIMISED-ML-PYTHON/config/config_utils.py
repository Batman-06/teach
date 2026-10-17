import re
from config.state_config import STATE_CLASS_CONFIG

def camel_to_upper_snake(name: str) -> str:
    """Convert camelCase or mixedCase names to UPPER_SNAKE_CASE"""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).upper()

def reset_to_class_defaults(state_class, attr_name):
    """
    Return the default value defined on the class (safe for mutable types).
    Works whether `state_class` is a class or an instance.
    """
    # Handle both class and instance cases
    cls = state_class if isinstance(state_class, type) else state_class.__class__

    # Try to get the class-level default
    default_value = getattr(cls, attr_name, None)

    if default_value is not None:
        if isinstance(default_value, dict):
            return {}
        elif isinstance(default_value, list):
            return []
        return default_value  # int, float, str, etc.

    # Fallback: look at current instance value (if state_class is an instance)
    if not isinstance(state_class, type):
        current_value = getattr(state_class, attr_name, None)
        if isinstance(current_value, dict):
            return {}
        elif isinstance(current_value, list):
            return []
        elif isinstance(current_value, (int, float)):
            return 0
        return current_value

    # Default fallback
    return None


def update_state_config(state_name: str, updates: dict):
    state_class = STATE_CLASS_CONFIG.get(state_name)
    if not state_class:
        raise ValueError(f"Invalid state: {state_name}")

    # Get all configurable attributes (non-private, non-methods)
    configurable_attrs = [
        attr for attr in dir(state_class)
        if not attr.startswith('_') and not callable(getattr(state_class, attr))
    ]

    # Flatten nested fixedCharges if present
    charges = updates.get("fixedCharges", {})
    flattened_updates = {**updates, **charges}
    flattened_updates.pop("fixedCharges", None)

    # Reset all configurable attributes to defaults
    for attr in configurable_attrs:
        if hasattr(state_class, attr):
            setattr(state_class, attr, reset_to_class_defaults(state_class, attr))

    # Apply payload updates
    for key, value in flattened_updates.items():
        attr = camel_to_upper_snake(key)
        if hasattr(state_class, attr) and value is not None:
            setattr(state_class, attr, value)
            # print(f"Updated values: {key} ➝ {value}")
        # else:
        #     # print(f"⚠️ The key from BE '{key}' is not found in state_config for state ➝ {state_name} as '{attr}'")
        #     pass
 

