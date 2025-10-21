import ast
import json


def validate_dsl(dsl_string: str, registry: dict) -> (dict, list):
    """
    Validates the DSL from the LLM against the feature registry.
    Also applies default values for missing parameters.
    Returns the enriched DSL with defaults filled in.
    """
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
        feature_name = feature_req.get("name")
        feature_parameters = feature_req.get("params", {})

        # Check if this is a custom feature (starts with 'custom_' prefix)
        is_custom_feature = feature_name and feature_name.startswith("custom_")

        if is_custom_feature:
            # Validate custom feature parameters
            if "code" not in feature_parameters:
                errors.append(
                    f"Feature {i} ('{feature_name}'): Missing required 'code' parameter for custom feature."
                )
            else:
                # Validate that the code is syntactically valid Python
                code = feature_parameters["code"]
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors.append(
                        f"Feature {i} ('{feature_name}'): Invalid Python syntax in code: {str(e)}"
                    )

            if "as" not in feature_parameters:
                errors.append(
                    f"Feature {i} ('{feature_name}'): Missing required 'as' parameter for output column name."
                )

            # No need to check registry params for custom features
            continue

        if feature_name not in registry["features"]:
            errors.append(f"Feature {i} ('{feature_name}'): Not a supported feature.")
            continue

        registry_params = registry["features"][feature_name].get("params", {})

        # Check for required parameters and apply defaults
        for (
            registry_parameter_name,
            registry_parameter_rules,
        ) in registry_params.items():
            parameter_required = registry_parameter_rules.get("required", False)
            parameter_has_default_val = "default" in registry_parameter_rules

            # Check if required parameter is missing
            if (
                parameter_required
                and registry_parameter_name not in feature_parameters
                and not parameter_has_default_val
            ):
                errors.append(
                    f"Feature {i} ('{feature_name}'): Required parameter '{registry_parameter_name}' is missing and has no default value."
                )
                continue

            # Apply default value if parameter is missing
            if (
                registry_parameter_name not in feature_parameters
                and parameter_has_default_val
            ):
                feature_parameters[registry_parameter_name] = registry_parameter_rules[
                    "default"
                ]

        # Validate provided parameters
        for parameter_name, parameter_value in feature_parameters.items():
            if parameter_name not in registry_params:
                errors.append(
                    f"Feature {i} ('{feature_name}'): Parameter '{parameter_name}' is not supported for this feature."
                )
                continue

            registry_parameter_rules = registry_params[parameter_name]
            expected_type = registry_parameter_rules.get("type")

            # Type check
            if expected_type == "string" and not isinstance(parameter_value, str):
                errors.append(
                    f"Feature {i} ('{feature_name}'): Parameter '{parameter_name}' must be a string, but got {type(parameter_value).__name__}."
                )
                continue
            elif expected_type == "int" and not isinstance(parameter_value, int):
                errors.append(
                    f"Feature {i} ('{feature_name}'): Parameter '{parameter_name}' must be an integer, but got {type(parameter_value).__name__}."
                )
                continue

            # Check allowed values
            if "allowed" in registry_parameter_rules:
                allowed_values = registry_parameter_rules["allowed"]
                if parameter_value not in allowed_values:
                    errors.append(
                        f"Feature {i} ('{feature_name}'): Parameter '{parameter_name}' has value '{parameter_value}', "
                        f"but must be one of {allowed_values}."
                    )

        # Update the feature request with the enriched params (including defaults)
        feature_req["params"] = feature_parameters

    if errors:
        return None, errors

    return dsl, []
