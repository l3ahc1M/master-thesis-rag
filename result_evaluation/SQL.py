import re
from result_evaluation.evaluation_metrics.exact_matching import exact_matching
from result_evaluation.evaluation_metrics.component_matching import component_matching

sql_components = [
    'select',
    'from',
    ('join', 'inner join', 'left join', 'right join', 'full join', 'cross join'),
    'on',
    'where',
    'group by',
    'having',
    'order by',
    'limit',
]

def _coerce_sql(value, *, label):
    """
    Try to coerce various shapes (str, dict with 'sql'/'query'/'text', list/tuple)
    into a SQL string. Raise a helpful error if not possible.
    """
    if value is None:
        raise ValueError(f"{label} is None; expected a SQL string.")

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for k in ('sql', 'query', 'text'):
            if k in value and isinstance(value[k], str):
                return value[k]
        raise ValueError(f"{label} looks like a dict without a 'sql'/'query'/'text' string")

    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))

    return str(value)


def remove_sql_comments(sql_text: str) -> str:
    """
    Remove SQL comments:
      - line comments starting with --
      - block comments /* ... */
    Keeps string/identifier literals intact.
    """
    i = 0
    length = len(sql_text)
    output = []
    inside_single_quote = False
    inside_double_quote = False
    inside_line_comment = False
    inside_block_comment = False

    while i < length:
        char = sql_text[i]
        next_char = sql_text[i+1] if i + 1 < length else ''

        # Handle line comment
        if inside_line_comment:
            if char == '\n':
                inside_line_comment = False
                output.append(char)
            i += 1
            continue

        # Handle block comment
        if inside_block_comment:
            if char == '*' and next_char == '/':
                inside_block_comment = False
                i += 2
            else:
                i += 1
            continue

        # Inside single-quoted string
        if inside_single_quote:
            output.append(char)
            if char == "'" and next_char == "'":  # escaped quote
                output.append(next_char)
                i += 2
                continue
            if char == "'":
                inside_single_quote = False
            i += 1
            continue

        # Inside double-quoted identifier
        if inside_double_quote:
            output.append(char)
            if char == '"' and next_char == '"':  # escaped double quote
                output.append(next_char)
                i += 2
                continue
            if char == '"':
                inside_double_quote = False
            i += 1
            continue

        # Detect new comments
        if char == '-' and next_char == '-':
            inside_line_comment = True
            i += 2
            continue
        if char == '/' and next_char == '*':
            inside_block_comment = True
            i += 2
            continue

        # Detect start of string literals
        if char == "'":
            inside_single_quote = True
            output.append(char)
            i += 1
            continue
        if char == '"':
            inside_double_quote = True
            output.append(char)
            i += 1
            continue

        # Normal character
        output.append(char)
        i += 1

    return ''.join(output)

def normalize_sql(sql_text: str) -> str:
    """
    Normalize SQL so two statements can be compared fairly:
      1) Remove comments
      2) Trim leading/trailing whitespace
      3) Drop trailing semicolons
      4) Convert to lowercase
      5) Replace newlines with spaces
      6) Collapse multiple spaces to single spaces
    """
    cleaned = remove_sql_comments(sql_text)
    cleaned = cleaned.strip()
    cleaned = re.sub(r';+\s*$', '', cleaned)  # strip trailing semicolons
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("\n", " ") # replace newlines with spaces
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " " ) # replace double spaces with single spaces
    return cleaned


def evaluate_sql_test_case(test_case):
    desired_result_raw = test_case['output']
    test_result_raw = test_case['test_output']

    desired_result = _coerce_sql(desired_result_raw, label="test_case['output']")
    test_result = _coerce_sql(test_result_raw, label="test_case['test_output']")

    normalized_desired = normalize_sql(desired_result)
    normalized_test = normalize_sql(test_result)


    test_case['component_matching_results'] = component_matching(normalized_desired, normalized_test, sql_components)

    test_case['exact_match'] = exact_matching(test_case['component_matching_results'])

    return test_case