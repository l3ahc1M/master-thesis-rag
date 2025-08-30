import re
from typing import Dict, List, Set, Tuple


def extract_clause(sql: str, start_keyword: str, end_keywords: List[str]) -> str:
    """
    Extract substring between `start_keyword` and the earliest of `end_keywords`.
    Returns empty string if not found.
    """
    start_match = re.search(rf"\b{start_keyword}\b", sql, flags=re.IGNORECASE)
    if not start_match:
        return ""

    start_index = start_match.end()
    end_candidates = []

    for keyword in end_keywords:
        end_match = re.search(rf"\b{keyword}\b", sql[start_index:], flags=re.IGNORECASE)
        if end_match:
            end_candidates.append(start_index + end_match.start())

    end_index = min(end_candidates) if end_candidates else len(sql)
    return sql[start_index:end_index].strip()

import re

import re

def normalize_quotes_and_commas(value: str) -> str:
    """
    Normalize quoted values by:
    1. Removing surrounding single or double quotes.
    2. Removing any commas and extra spaces.
    3. Removing any leading colons (':').
    Iterates until no changes are made.
    """
   
    previous_value = None
    while value != previous_value:
        previous_value = value  # Store the current value to detect changes
        
        # Remove leading colons if present
        value = value.lstrip(":").strip()
        
        # Remove surrounding single or double quotes
        value = re.sub(r"^['\"](.*)['\"]$", r"\1", value)
        
        # Remove any trailing commas and extra spaces
        value = value.rstrip(",").strip()
        
        # Remove any extra spaces around colons (if any still exist)
        value = re.sub(r"\s*[:]\s*", ":", value)  # Normalize space around colons

    
    return value



def get_component_content(text, components):
    i = 0
    components_content = []
    text = (text + "[END]")  # sentinel to capture last clause
    components.append("[END]")  # append sentinel to handle last clause

    for component in components:
        clause = extract_clause(text, component, components)
        components_content.append({
            'Component': component,
            'Clause': normalize_quotes_and_commas(clause)
        })
        
    return components_content




def component_matching(predicted, test_result, components):
    predicted_components = get_component_content(predicted, components)
    test_result_components = get_component_content(test_result, components)

    component_evaluation = []

    # lists are of the same length and correspond one-to-one.
    for predicted_component, test_result_component in zip(predicted_components, test_result_components):
        # Make sure each predicted and test result component has the expected structure.
        predicted_clause = predicted_component['Clause'].lower() if 'Clause' in predicted_component else ''
        test_result_clause = test_result_component['Clause'].lower() if 'Clause' in test_result_component else ''
        if len(predicted_clause) == 0:
            has_component = False
        else:
            has_component = True


        # Check for exact match between predicted and actual clauses
        evaluation = {
            'component': predicted_component['Component'],
            'has_component': has_component,
            'predicted_clause': predicted_clause,
            'test_result_clause': test_result_clause,
            'match': predicted_clause == test_result_clause
        }
        component_evaluation.append(evaluation)

    return component_evaluation

        