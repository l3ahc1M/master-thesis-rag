import os
import json
import yaml
from copy import deepcopy
from chroma_handler.ingestion import store_text_embedding


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg_path = os.path.join(root_dir, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    def get_api_json_files():
        api_json_files = []
        source_folder = cfg.get('system_documentation', {}).get('source_folder')
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.startswith("API_") and file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    # Get the subfolder name as business_object_name
                    rel_path = os.path.relpath(root, source_folder)
                    business_object_name = os.path.basename(rel_path) if rel_path != '.' else ''
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_content = json.load(f)
                        api_json_files.append({
                            'business_object_name': business_object_name,
                            'file_name': file,
                            'swagger_content': json_content
                        })
                    except Exception as e:
                        # Optionally log or handle error
                        pass
        return api_json_files


def chunk_swagger_documentation(swagger_content: dict, include_get: bool = False):
    """
    Produce per-endpoint chunks with NO information loss.

    For each path+method we return:
      {
        'http_method': 'POST' | ...,
        'path': '/pets/{id}',
        'path_item_raw': <original path item dict>,
        'operation_raw': <original operation dict>,
        'operation_resolved': <operation dict with all in-doc $ref recursively expanded>,
        'parameters': {
            'path_level_raw': [...],
            'path_level_resolved': [...],
            'operation_level_raw': [...],
            'operation_level_resolved': [...],
            'effective_parameters_resolved': [...]  # merged (operation overrides path-level by (in,name))
        },
        'referenced_components': {
            'schemas': [...], 'parameters': [...], 'responses': [...],
            'requestBodies': [...], 'headers': [...], 'links': [...],
            'examples': [...], 'callbacks': [...]
        }
      }

    Notes
    -----
    - Resolves only in-document refs ("#/..."). External refs are left as-is (kept in raw).
    - JSON Pointer unescaping (~0 -> ~, ~1 -> /) supported.
    - Cycle-safe: if a cycle is detected, leaves the $ref as-is at that point.
    - Absolutely no fields are dropped: we always keep the *raw* originals.
    """

    HTTP_METHODS = {'get','post','put','patch','delete','options','head','trace'}

    def _unescape_json_pointer(token: str) -> str:
        return token.replace('~1', '/').replace('~0', '~')

    def _resolve_pointer(ref: str, root: dict):
        if not isinstance(ref, str) or not ref.startswith('#/'):
            return None
        parts = [_unescape_json_pointer(p) for p in ref.lstrip('#/').split('/')]
        obj = root
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return deepcopy(obj) if isinstance(obj, (dict, list)) else obj

    # Track referenced component names by type
    def _new_ref_tracker():
        return {
            "schemas": set(), "parameters": set(), "responses": set(),
            "requestBodies": set(), "headers": set(), "links": set(),
            "examples": set(), "callbacks": set()
        }

    def _record_ref(ref: str, tracker: dict):
        try:
            parts = ref.lstrip('#/').split('/')
            if len(parts) >= 3 and parts[0] == 'components':
                comp_type, comp_name = parts[1], parts[2]
                if comp_type in tracker:
                    tracker[comp_type].add(comp_name)
        except Exception:
            pass

    def _resolve_refs_recursive(node, root, seen=None, tracker=None):
        """
        Deeply resolve {"$ref": "#/..."} dicts by replacement with target object.
        - Preserves unknown keys by resolving children as well.
        - Leaves external refs untouched.
        - Guard against cycles with `seen`.
        Returns (resolved_node, tracker)
        """
        if tracker is None:
            tracker = _new_ref_tracker()
        if seen is None:
            seen = set()

        if isinstance(node, dict):
            # If this dict is a pure $ref, replace the node entirely with the resolved target
            if set(node.keys()) == {"$ref"} and isinstance(node["$ref"], str) and node["$ref"].startswith("#/"):
                ref = node["$ref"]
                if ref in seen:
                    # cycle: leave as-is
                    return deepcopy(node), tracker
                target = _resolve_pointer(ref, root)
                if target is None:
                    # broken ref: keep as-is
                    return deepcopy(node), tracker
                seen.add(ref)
                _record_ref(ref, tracker)
                return _resolve_refs_recursive(target, root, seen, tracker)

            # Otherwise, resolve values; also handle nested $ref inside dicts
            out = {}
            for k, v in node.items():
                rv, tracker = _resolve_refs_recursive(v, root, seen, tracker)
                out[k] = rv
            return out, tracker

        if isinstance(node, list):
            out_list = []
            for item in node:
                rv, tracker = _resolve_refs_recursive(item, root, seen, tracker)
                out_list.append(rv)
            return out_list, tracker

        # primitives
        return deepcopy(node), tracker

    def _resolve_list(lst):
        if not isinstance(lst, list):
            return [], _new_ref_tracker()
        resolved = []
        tracker = _new_ref_tracker()
        for item in lst:
            r, t = _resolve_refs_recursive(item, swagger_content)
            resolved.append(r)
            for k in tracker:
                tracker[k] |= t[k]
        return resolved, tracker

    def _merge_params(path_params_resolved, op_params_resolved):
        """
        Merge per OAS rules: operation-level overrides path-level by (in, name).
        No loss: the originals are already kept separately.
        """
        def key(p):
            return (p.get("in"), p.get("name"))
        merged = []
        seen_keys = set()
        # Start with path-level, but they can be overridden later
        index = {key(p): deepcopy(p) for p in path_params_resolved if isinstance(p, dict)}
        # Apply/override with op-level
        for p in op_params_resolved:
            if isinstance(p, dict):
                index[key(p)] = deepcopy(p)
        # Preserve deterministic order: op-level first (more specific), then remaining path-level in original order
        for p in op_params_resolved:
            k = key(p)
            if k not in seen_keys:
                merged.append(deepcopy(index[k]))
                seen_keys.add(k)
        for p in path_params_resolved:
            k = key(p)
            if k not in seen_keys:
                merged.append(deepcopy(index[k]))
                seen_keys.add(k)
        return merged

    chunks = []
    paths = swagger_content.get('paths', {}) or {}
    if not isinstance(paths, dict):
        return chunks

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Keep the *raw* path item (no loss)
        path_item_raw = deepcopy(path_item)

        # Resolve path-level parameters (but keep the raw list too)
        path_level_params_raw = path_item.get('parameters', []) if isinstance(path_item.get('parameters'), list) else []
        path_level_params_resolved, path_param_refs = _resolve_list(path_level_params_raw)

        for method, operation in path_item.items():
            m = (method or "").lower()
            if m not in HTTP_METHODS:
                continue
            if m == 'get' and not include_get:
                continue
            if not isinstance(operation, dict):
                continue

            # Raw operation (no changes)
            operation_raw = deepcopy(operation)

            # Resolve operation *entirely* (no summarization)
            operation_resolved, op_refs = _resolve_refs_recursive(operation_raw, swagger_content)

            # Resolve operation-level parameters (and keep raw)
            op_params_raw = operation.get('parameters', []) if isinstance(operation.get('parameters'), list) else []
            op_params_resolved, op_param_refs = _resolve_list(op_params_raw)

            # Effective merged parameters (resolved view)
            effective_params_resolved = _merge_params(path_level_params_resolved, op_params_resolved)

            # Aggregate referenced components
            ref_tracker = _new_ref_tracker()
            for k in ref_tracker:  # union all trackers
                ref_tracker[k] |= path_param_refs[k]
                ref_tracker[k] |= op_param_refs[k]
                ref_tracker[k] |= op_refs[k]

            chunk = {
                'http_method': m.upper(),
                'path': path,
                'path_item_raw': path_item_raw,
                'operation_raw': operation_raw,
                'operation_resolved': operation_resolved,
                'parameters': {
                    'path_level_raw': path_level_params_raw,
                    'path_level_resolved': path_level_params_resolved,
                    'operation_level_raw': op_params_raw,
                    'operation_level_resolved': op_params_resolved,
                    'effective_parameters_resolved': effective_params_resolved,
                },
                'referenced_components': {k: sorted(v) for k, v in ref_tracker.items()},
            }

            chunks.append(chunk)

    return chunks

def preprocess_api_documentation():
    api_json_files = get_api_json_files()
    for api_file in api_json_files:
        business_object = api_file['business_object_name']
        file_name = api_file['file_name']
        swagger_content = api_file['swagger_content']
        print(f"Processing API file: {file_name} (Business Object: {business_object})")
        chunks = chunk_swagger_documentation(swagger_content)
        print(f"Created {len(chunks)} out of API file")
        for idx, chunk in enumerate(chunks):
            # Prepare text for embedding (can be customized as needed)
            source_type = "API"
            method = chunk.get('http_method')
            path = chunk.get('path')
            # Unique id for each chunk
            chunk_id = f"{file_name}_{method}_{path}_{idx}"

            #print(f"File name: {file_name}")
            #print(f"Method: {method}")
            #print(f"Path: {path}")
            
            store_text_embedding(
                text=str(chunk),
                file_name=file_name,
                business_object=business_object,
                source_type=source_type,
                id=chunk_id,
                method=method,
                path=path
            )
    