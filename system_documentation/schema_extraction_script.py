import json
import os


# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through all subdirectories in the script directory
for folder_name in os.listdir(script_dir):
    folder_path = os.path.join(script_dir, folder_name)

    # Only process if it's a directory
    if os.path.isdir(folder_path):
        # Look for a JSON file starting with "API_"
        api_files = [f for f in os.listdir(folder_path) if f.startswith("API_") and f.endswith(".json")]
        if not api_files:
            continue  # Skip this folder if no matching file is found

        input_path = os.path.join(folder_path, api_files[0])
        output_path = os.path.join(folder_path, f"DB_{folder_name}.json")

        # Try loading the API specification as JSON
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                api_spec = json.load(file)
        except Exception as e:
            print(f"Error loading JSON from {input_path}: {e}")
            continue

        result = {"tables": []}  # Structure for storing extracted table definitions

        paths = api_spec.get("paths", {})  # All endpoint paths from the API spec
        schemas = api_spec.get("components", {}).get("schemas", {})  # All defined object schemas
        processed_refs = set()  # Track references that have already been processed

        # Iterate through all paths in the API
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() == "get":  # Only consider GET methods
                    responses = details.get("responses", {})
                    for response in responses.values():
                        content = response.get("content", {})
                        for media_type in content.values():
                            schema_data = media_type.get("schema", {})
                            # Get schema reference from "items" or directly from schema
                            schema_ref = schema_data.get("items", {}).get("$ref") or schema_data.get("$ref")

                            if schema_ref:
                                ref_name = schema_ref.split("/")[-1]  # Extract schema name
                                if ref_name in processed_refs:
                                    continue  # Skip if already processed
                                processed_refs.add(ref_name)
                                schema = schemas.get(ref_name)

                                # Only process object-type schemas that have properties
                                if schema and schema.get("type") == "object" and "properties" in schema:
                                    table = {
                                        "name": ref_name.split(".")[-1],
                                        "description": schema.get("description", "").strip(),
                                        "columns": []
                                    }

                                    # Iterate through each property in the schema
                                    for prop_name, prop in schema["properties"].items():
                                        # Add example value to the description if available
                                        base_description = prop.get("title", prop.get("description", "")).strip()
                                        example = prop.get("example")
                                        if example is not None:
                                            full_description = f"{base_description} Example: {example}"
                                        else:
                                            full_description = base_description

                                        column = {
                                            "name": prop_name,
                                            "description": full_description,
                                            "format": prop.get("format", prop.get("type", "string"))
                                        }

                                        # Check if the property references another schema that contains enums
                                        if "$ref" in prop:
                                            nested_ref = prop["$ref"].split("/")[-1]
                                            nested_schema = schemas.get(nested_ref, {})
                                            if "enum" in nested_schema:
                                                column["format"] = "enum"
                                                column["enum_values"] = nested_schema["enum"]

                                        # Or if the enum is directly defined in the property
                                        elif "enum" in prop:
                                            column["format"] = "enum"
                                            column["enum_values"] = prop["enum"]

                                        table["columns"].append(column)  # Add column to the table

                                    result["tables"].append(table)  # Add table to the result list

        # Write the extracted structure to the output file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"Error writing file {output_path}: {e}")
