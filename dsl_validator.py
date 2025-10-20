import json


def validate_dsl(dsl_string: str, registry: dict) -> (dict, list):
    # Validates the DSL  from the LLM against the feature registry.
    errors = []
    try:
        dsl = json.loads(dsl_string)
    except json.JSONDecodeError:
        errors.append("Validation Error: LLM output was not valid JSON.")
        return None, errors

    if "features" not in dsl or not isinstance(dsl["features"], list):
        errors.append("Validation Error: JSON must have a top-level 'features' key.")
        return None, errors

    for i, feature_req in enumerate(dsl.get("features", [])):
        name = feature_req.get("name")
        params = feature_req.get("params", {})

        if name not in registry["features"]:
            errors.append(f"Feature {i} ('{name}'): Not a supported feature.")
            continue

        registry_params = registry["features"][name].get("params", {})
        # TODO: add support for 'allowed' field
        # TODO: add support for 'required' field
        # TODO: add support for loading 'default' field if field not found
        for p_name, p_value in params.items():
            if p_name in registry_params:
                expected_type = registry_params[p_name].get("type")

                # type check
                if expected_type == "string" and not isinstance(p_value, str):
                    errors.append(
                        f"Feature {i} ('{name}'): Parameter '{p_name}' must be a string, but got {type(p_value)}."
                    )
                elif expected_type == "int" and not isinstance(p_value, int):
                    errors.append(
                        f"Feature {i} ('{name}'): Parameter '{p_name}' must be an integer, but got {type(p_value)}."
                    )

    if errors:
        return None, errors

    return dsl, []
