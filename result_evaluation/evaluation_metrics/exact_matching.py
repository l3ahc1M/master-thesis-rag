def exact_matching(component_matching_results):
    num_errors = 0
    
    
    if not isinstance(component_matching_results, list):
        raise TypeError("Expected a list of dictionaries, but got something else.")

    for component in component_matching_results:
        # Check if each component is a dictionary and contains 'match'
        if isinstance(component, dict):
            if 'match' not in component:
                raise KeyError("'match' key not found in component dictionary.")
            if component['match'] is False:
                num_errors += 1
        else:
            raise TypeError(f"Expected component to be a dictionary, but got {type(component)}")

    return num_errors == 0  # Returns True if no errors, False otherwise
