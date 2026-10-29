# COUNTSCHECK: reduce the OpenAI-usage scan rows (.rows[], path = parts joined by "/") and the usage record (.runs[],
# .path) to one bit per model name: "some token count > 0". No token count leaves jq.
def bits: (. // {}) | with_entries(.value |= ([.[]? | numbers | select(. > 0)] | length > 0));
if has("rows") then [.rows[] | {path: (.parts | join("/")), models: (.model_usage | bits)}]
else [.runs[] | {path, models: (.usage | bits)}] end
