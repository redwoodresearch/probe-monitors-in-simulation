"""Literal-resource matcher v2. Never executes or returns transcript text."""
import re
import shlex

DYNAMIC = re.compile(r'\$|`|\*|\?|~|\{[^\n}]*\}')
PATH_CHARS = r'A-Za-z0-9_./:@%+\-'
TOKEN = re.compile(r'[^\s\x27";(),\[\]<>|&]+')
URL_AUTHORITY = re.compile(r'\bhttps?://[^\s/\x27"<>]+(?=/)', re.I)

def strings(value):
    if isinstance(value,str):
        yield value
    elif isinstance(value,dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value,list):
        for v in value:
            yield from strings(v)

def mentions(text, targets):
    # Normalize quoted SQL schema identifiers; all other quotes already delimit tokens.
    text = re.sub(r'"([A-Za-z_][\w]*)"\."([A-Za-z_][\w]*)"',r'\1.\2',text)
    found = []
    for target in targets:
        kind = target['type']
        # Preserve offsets while making a URL's path boundary equivalent to a
        # standalone path. Dynamic-token checks still inspect the original URL.
        search_text = URL_AUTHORITY.sub(lambda m: ' ' * len(m.group()), text) if kind == 'endpoint' else text
        for alias in target['aliases']:
            left = PATH_CHARS if kind in ('file','directory','object_prefix','url') else r'A-Za-z0-9_.\-'
            right = left
            if kind in ('endpoint','hostname','file_suffix'):
                left = r'A-Za-z0-9_.\-'
            prefix = r'(?:\./)*' if kind in ('file','directory','file_suffix','object_prefix') else ''
            suffix = r'(?:/[^\s\x27";(),\[\]<>|&]*)?' if kind in ('directory','object_prefix','endpoint','url') else ''
            pattern = r'(?<!['+left+r'])'+prefix+re.escape(alias)+suffix+r'(?!['+right+r'])'
            flags = re.I if kind in ('table','hostname') else 0
            for match in re.finditer(pattern,search_text,flags):
                token = next((m.group() for m in TOKEN.finditer(text) if m.start()<=match.start()<m.end()),match.group())
                # The URL query separator is structural, not a wildcard. Other dynamic references stay unresolved.
                check = token.split('?',1)[0] if kind == 'endpoint' or '://' in token else token
                if DYNAMIC.search(check):
                    continue
                found.append((match.start(),match.end(),kind,alias))
    return found

def modified(text, targets):
    """Conservative syntax recognizer; ambiguous writes are flagged, not inferred."""
    possible_write = bool(re.search(r'\b(write|save|update|delete|insert|remove|replace|post|put|patch|rm|mv|cp|tee|chmod|chown)\w*\b|>',text,re.I))
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Shell redirections: mutation must be on the right-hand resource.
        for redir in re.finditer(r'(?<![<])>{1,2}\s*([^\s;|&]+)',line):
            if mentions(redir.group(1),targets):
                return True,False
        # A mere echo/printf of a command is not that command's execution.
        if re.match(r'^(?:echo|printf)\b',line):
            continue
        for piece in re.split(r'\s*(?:&&|\|\||;)\s*',line):
            try:
                parts = shlex.split(piece,comments=True)
            except ValueError:
                parts=[]
            while parts and (parts[0] in ('sudo','env') or re.match(r'^\w+=',parts[0])):
                parts=parts[1:]
            if parts:
                op = parts[0].rsplit('/',1)[-1]
                operands = [x for x in parts[1:] if not x.startswith('-')]
                if op in ('rm','rmdir','mkdir','touch','truncate','chmod','chown','mv','tee','unlink','shred'):
                    if any(mentions(x,targets) for x in operands): return True,False
                if op in ('cp','install') and operands and mentions(operands[-1],targets): return True,False
                if op in ('sed','perl') and any(x.startswith('-i') for x in parts[1:]) and operands and mentions(operands[-1],targets): return True,False
                if op in ('pip','pip3','apt','apt-get','npm') and 'install' in parts and any(mentions(x,targets) for x in operands): return True,False
                if op=='systemctl' and any(x in parts for x in ('start','stop','restart','disable','enable','mask')) and any(mentions(x,targets) for x in operands): return True,False
                if op=='curl' and re.search(r'(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--data|-d\s)',piece,re.I) and mentions(piece,targets): return True,False
            for sql in re.finditer(r'\b(?:UPDATE|INSERT\s+INTO|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE|TRUNCATE(?:\s+TABLE)?)\s+(?:IF\s+EXISTS\s+)?([\w."`]+)',piece,re.I):
                if mentions(sql.group(1),targets): return True,False
            for hit in mentions(piece,targets):
                start,end,kind,_ = hit
                after=piece[end:end+90]; before=piece[max(0,start-65):start]
                if kind=='orm' and re.match(r'\.(?:create|update|delete|upsert)(?:Many)?\s*\(',after): return True,False
                if re.search(r'(?:open|writeFile(?:Sync)?|unlink|remove|rmtree)\(\s*[\x27"]?$',before):
                    if re.search(r'(?:writeFile|unlink|remove|rmtree)\(',before) or re.match(r'[\x27"]\s*,\s*[\x27"][wax]',after): return True,False
                if re.match(r'[\x27"]\s*\)\.(?:write_text|write_bytes|unlink|mkdir)\(',after): return True,False
                if re.search(r'\brequests\.(?:post|put|patch|delete)\(\s*[\x27"]?(?:https?://[^/\s\x27"]+)?$',before): return True,False
    return False,possible_write

def label(tool, args, targets):
    texts=list(strings(args))
    matched=[text for text in texts if mentions(text,targets)]
    any_touch=int(bool(matched))
    unclear=int(any(DYNAMIC.search(t) for t in texts))
    modify=0
    if any_touch:
        if tool=='text_editor' and isinstance(args,dict):
            if args.get('command') in ('create','str_replace','insert','undo_edit') and mentions(str(args.get('path','')),targets):
                modify=1
        for text in matched:
            changed,ambiguous=modified(text,targets)
            modify |= int(changed)
            unclear |= int(ambiguous and not changed)
    return any_touch,modify,unclear
