# 09 S1b-7 PSR, step 1: KEY NAMES ONLY (no values). Top-level keys, meta's keys, and the key names of gate-like objects.
{top: keys,
 meta_keys: ((.meta // {}) | keys),
 gate_keys: ((.gate // {}) | if type == "object" then keys else type end),
 meta_nested_keys: ((.meta // {}) | with_entries(select(.value | type == "object")) | map_values(keys))}
