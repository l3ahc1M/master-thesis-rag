
import re
from result_evaluation.evaluation_metrics.exact_matching import exact_matching
from result_evaluation.evaluation_metrics.component_matching import component_matching

api_components = [
    'method',
    'endpoint',
    'body'
]

def normalize_api(api_text: str) -> str:
    """
    Normalize SQL so two statements can be compared fairly:
      1) Trim leading/trailing whitespace
      2) Drop trailing semicolons
      3) Convert to lowercase
      4) Replace newlines with spaces
      5) Collapse multiple spaces to single spaces
    """
    cleaned = str(api_text).strip()
    cleaned = re.sub(r';+\s*$', '', cleaned)  # strip trailing semicolons
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("\n", " ") # replace newlines with spaces
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " " ) # replace double spaces with single spaces
    return cleaned

def evaluate_api_test_case(test_case):
    desired_result = test_case['output']
    test_result = test_case['test_output']

    normalized_desired = normalize_api(desired_result)
    normalized_test = normalize_api(test_result)
    
    test_case['component_matching_results'] = component_matching(normalized_desired, normalized_test, api_components)

    test_case['exact_match'] = exact_matching(test_case['component_matching_results'])

    return test_case