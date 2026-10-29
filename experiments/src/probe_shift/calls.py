"""Label each tool call read-only or impactful (not read-only), by rules on the command text.

`label_call(tool, arguments)` returns the label, the call's primary category, every category found and the rules
that fired. A shell command is split into simple commands (outside quotes; a heredoc body goes with the command that
reads it; `$(...)` contents are labeled on their own) and each piece gets a category. A call is read-only only if
every piece is clearly read-only; anything else is impactful (Aswin, 2026-10-06):

- read-only categories: db_read, inline_code_read, file_read, inspect;
- impactful, clear: file_delete, db_write, process_service, install, network, permissions_config, file_write,
  file_write_tmp (writes whose only targets are under /tmp still count);
- impactful, unclear: run_program (scripts, builds, tests, inline code that loads project code: side effects not
  visible in the command) and unknown (no rule matched). The category column keeps this switchable; analyses
  report a robustness row without these calls.

Every network call is impactful (category network), whatever its method or destination (Aswin, 2026-10-06): curl,
wget, ssh, nc, aws, vault, kubectl, HTTP and other clients in inline code, and database clients given a host
(`psql -h db`). Files curl and wget save are also writes. Each call carries net_method, net_dest and net_sends_data
(None without a network piece), so a narrower network rule can be recomputed without relabeling.

Inline code (`python3 -c`, a heredoc fed to an interpreter, `node -e`, ...) is scanned for writes, deletes, network,
process control and SQL writes.
"""

import ipaddress
import re
import shlex
import sys

CLEAR_IMPACTFUL = ["file_delete", "db_write", "process_service", "install", "network", "permissions_config",
                   "file_write", "file_write_tmp"]
UNCLEAR = ["run_program", "unknown", "malformed"]  # impactful, but only because their effects are not visible
IMPACTFUL = CLEAR_IMPACTFUL + UNCLEAR
READ_ONLY = ["db_read", "inline_code_read", "file_read", "inspect"]
PRIORITY = IMPACTFUL + READ_ONLY  # a call's primary category is its first category in this order
HELPER_READ = re.compile(r"(^|[-_])(get|check|list|history|show|validate|preview)([-_]|$)")  # helpers named for reading

SHELL_TOOLS = {"bash", "db_bash", "client_bash", "rust_worker_bash", "py_ml_bash", "nginx_bash"}
EDITOR_WRITES = {"create", "str_replace", "insert", "undo_edit"}


def _table(**groups: str) -> dict[str, str]:
    return {word: cat for cat, words in groups.items() for word in words.split()}


# Simple commands whose category does not depend on their arguments.
SIMPLE = _table(
    inspect="cd pushd popd echo printf true false : sleep wait set unset export local declare typeset readonly read "
            "shift trap exit return break continue alias ulimit umask hash test [ [[ ]] let fi done esac readarray mapfile "
            "for select case function in shopt clear tput seq yes ps pgrep pidof ss netstat lsof id whoami groups "
            "printenv uname hostname date df free uptime journalctl top nproc dpkg-query apt-cache which type whereis ip "
            "ifconfig locale tty lscpu lsblk vmstat w who last dmesg pwd findmnt arch getconf nvidia-smi lsmod pstree "
            "fuser pg_isready pg_config pkg-config man info timedatectl loginctl namei envsubst aa-status getenforce "
            "sestatus pg_lsclusters route",
    file_read="cat head tail less more grep egrep fgrep rg ag ls tree stat wc diff cmp uniq cut tr nl xxd od hexdump "
              "strings md5sum sha1sum sha256sum sha512sum cksum readlink realpath dirname basename column comm join "
              "paste fold fmt rev tac zcat zgrep bzcat xzcat file expand iconv ldd nm objdump readelf lsattr getfacl "
              "du col base64 jq javap",
    network="wget nc ncat netcat telnet ssh scp sftp ftp http ping dig nslookup host traceroute aws gcloud "
            "gsutil socat grpcurl websocat lftp smbclient ldapsearch ldapadd ldapmodify ldapdelete mosquitto_pub "
            "kubectl rclone ssh-keyscan nmap",
    file_write="cp mv mkdir touch ln install dd tee patch zip mktemp split mkfifo ssh-keygen htpasswd logger",
    file_delete="rm rmdir unlink shred truncate",
    process_service="kill pkill killall nohup setsid disown start-stop-daemon redis-server gunicorn uvicorn celery "
                    "reboot shutdown rsyslogd pgbouncer postgres mysqld mongod etcd rabbitmq-server sshd crond cron "
                    "screen tmux at puma sidekiq unicorn thin foreman hypercorn daphne",
    permissions_config="chmod chown chgrp setfacl chattr useradd userdel usermod groupadd groupdel groupmod adduser "
                       "deluser addgroup delgroup passwd chpasswd visudo iptables ip6tables ufw update-ca-certificates "
                       "setcap chsh gpasswd chage locale-gen hostnamectl modprobe umount chroot",
    db_write="createdb dropdb createuser dropuser pg_restore mysqlimport",
    run_program="make cmake ninja meson gcc g++ cc c++ clang clang++ javac java mvn gradle rustc tsc pytest tox nox "
                "rails rake rspec phpunit jest mocha vitest sbt swift ctest valgrind gdb ab wrk hey locust k6 expect "
                "jupyter papermill alembic prisma terraform ansible-playbook helm vim vi nano emacs mc gpg aide mypy "
                "pylint flake8 shellcheck golangci-lint cppcheck clang-tidy bats ffmpeg clojure lein jar run-parts next vite webpack "
                "logrotate",
)
WRAPPERS = {"sudo", "timeout", "nice", "ionice", "stdbuf", "time", "flock", "runuser", "doas", "unbuffer", "gosu",
            "su-exec", "tini", "env", "nohup", "setsid", "exec", "command", "xargs", "watch", "strace", "builtin"}
WRAPPER_OPTS_WITH_VALUE = {"-u", "-g", "--user", "-n", "-P", "-I", "-L", "-d", "-s", "-c", "-w", "-C", "-k", "-p"}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
INTERPRETERS = {"python", "node", "ruby", "perl", "Rscript", "R", "php", "deno", "bun", "ts-node", "tsx", "lua", "julia"}
# Per interpreter: options that take inline code, options that only check syntax, options whose value is the next
# word (or attached, as in -rjson).
INTERPRETER_OPTS = {
    "python": ({"-c"}, set(), {"-W", "-X"}),
    "node": ({"-e", "--eval", "-p", "--print"}, {"-c", "--check"}, {"-r", "--require", "--import", "--loader"}),
    "ruby": ({"-e"}, {"-c"}, {"-r", "-I", "-C", "-E", "-K"}),
    "perl": ({"-e", "-E"}, {"-c"}, {"-I", "-M", "-m"}),
    "php": ({"-r"}, {"-l"}, {"-d", "-c", "-f"}),
    "R": ({"-e"}, set(), {"-f"}),
}
TCP_BY_DEFAULT = {"redis-cli", "mongosh", "mongo", "etcdctl"}  # unlike psql and mysql, no local socket by default
DB_CLIENTS = {"psql", "mysql", "mariadb", "sqlite3", "duckdb", "mongosh", "mongo", "redis-cli", "etcdctl"}
PG_TOOLS = {"psql", "pg_dump", "pg_dumpall", "pg_restore", "pg_isready", "createdb", "dropdb", "createuser", "dropuser"}
# Database tools that go over the network when given a host (-h, --host, a connection URL); see _db_host.
DB_HOST_TOOLS = DB_CLIENTS - {"sqlite3", "duckdb"} | PG_TOOLS | {"mysqldump", "mysqladmin", "mysqlimport", "mongodump",
                                                                 "mongorestore", "mongoexport", "mongoimport"}
VERSION_ARGS = {"--version", "-version", "-V", "-v", "version", "--help", "-h", "help", "--info"}
VERBOSE_V = {"pytest", "py.test", "tox"}  # -v: a verbose run (jest, mocha, vitest and rspec read it as --version)
CONFIG_TEST = {"sshd": {"-t", "-T"}, "rsyslogd": {"-N1", "-N2", "-N3"}, "apache2": {"-t"}, "httpd": {"-t"},
               "haproxy": {"-c"}}  # check the configuration and exit
SERVICE_NAME = re.compile(r"(restart|start|stop|reload|deploy|launch|serve|daemon)", re.I)
DEV_NULL = re.compile(r"^/dev/(null|stdout|stderr|tty|fd/[0-2])$")
REDIRECT = re.compile(r"(?<![<>&=-])(\d*|&)(>>|>\||>)(?!&)(?!\()\s*([^\s;&|<>()]+)")
OTHER_REDIRECT = re.compile(r"<{2,3}-?\s*\S+|\d*<>\s*\S+|\d*[<>]&\d*-?|&>>?\s*\S+|\d*<\s*[^\s;&|<>]+")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
CONTROL = {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}", "time"}

CODE_RULES = [
    ("file_delete", r"os\.(remove|unlink|rmdir)\(|\.unlink\(|shutil\.rmtree|\.rmdir\(|fs\.(unlink|rm|rmdir)(Sync)?\(|"
                    r"File\.delete|FileUtils\.rm|file\.remove\(|\bunlink\("),
    ("install", r"install\.packages\(|(devtools|remotes)::install_\w+\(|BiocManager::install\(|pip\.main\("),
    ("process_service", r"os\.kill\(|process\.kill\(|Process\.kill|\.terminate\(\)|\bpkill\b|systemctl"),
    ("run_program", r"subprocess|os\.(system|popen)\(|Popen|child_process|execSync|spawnSync|\bspawn\(|%x\(|"
                    r"\bsystem\(|shell_exec|passthru|proc_open|Open3"),
    ("file_write", r"write_(text|bytes)\(|open\([^)]*,\s*['\"][wax]b?\+?['\"]|open\([^)]*mode\s*=\s*['\"][wax]|(?<!stdout)(?<!stderr)(?<!STDOUT)(?<!STDERR)\.write\(|"
                   r"json\.dump\(|pickle\.dump\(|shutil\.(copy|move|copyfile|copytree)|os\.(rename|replace|makedirs|"
                   r"mkdir|symlink|chmod|chown)\(|\.(mkdir|touch|rename)\(|fs\.(writeFile|appendFile|mkdir|copyFile|"
                   r"rename|chmod|createWriteStream)(Sync)?\(|File\.(write|open\([^)]*['\"][wa])|IO\.write|FileUtils\.|"
                   r"writeLines\(|saveRDS\(|sink\(|file_put_contents|fwrite\(|fputs\(|"
                   r"savefig\(|np\.save|torch\.save|dbWriteTable|file\.(copy|rename)\(|"
                   r"dir\.create\("),
]
SQL_WRITE = re.compile(
    r"\b(INSERT\s+INTO|UPDATE\s+[\w\".]+\s+SET|DELETE\s+FROM|(DROP|ALTER)\s+(TABLE|INDEX|DATABASE|SCHEMA|VIEW|"
    r"FUNCTION|TRIGGER|USER|ROLE|SEQUENCE|TYPE|EXTENSION|POLICY|MATERIALIZED)|CREATE\s+(OR\s+REPLACE\s+)?(UNIQUE\s+|"
    r"TEMP\s+|TEMPORARY\s+)?(TABLE|INDEX|DATABASE|SCHEMA|VIEW|FUNCTION|TRIGGER|USER|ROLE|SEQUENCE|TYPE|EXTENSION|"
    r"PROCEDURE|MATERIALIZED|POLICY|RULE)|TRUNCATE\b|GRANT\s|REVOKE\s|COPY\s+[\w\".]+\s+FROM|VACUUM|REINDEX|COMMENT\s+ON|REFRESH\s+MATERIALIZED|"
    r"setval|pg_terminate_backend|LOCK\s+TABLE)", re.I)
DB_IN_CODE = re.compile(r"psycopg|sqlite3|PDO|\bpg\b|mysql|PrismaClient|redis|DBI::|dbConnect|PQconnect|sql\.Open", re.I)
ORM_WRITE = re.compile(r"\.(create|createMany|update|updateMany|upsert|delete|deleteMany)\(\s*\{|\.(save|destroy|"
                       r"update_all|delete_all|insert_one|insert_many|update_one|update_many|delete_one|delete_many)\(")
PROJECT_CODE = re.compile(r"\bsource\(|require\(\s*['\"]\.{0,2}/|from\s+['\"]\.{0,2}/|import\(\s*['\"]\.|require_relative|"
                          r"\bload\s+['\"]|require\s+['\"]\.{0,2}/|\b(require|include)(_once)?\s*\(?\s*['\"]/|sys\.path\.|"
                          r"importlib|Rails\.|ActiveRecord")
WRITE_TARGET = re.compile(r"(?:open|Path|writeFileSync|writeFile|appendFileSync|File\.write|file_put_contents)\(\s*['\"]/|"
                          r"writeLines\([^,]+,\s*(?:con\s*=\s*)?['\"]/")
STRINGS = re.compile(r"(\'\'\'|\"\"\")[\s\S]*?\1|`[^`]*`|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"")
MODULE_NAME = re.compile(r"(?:\brequire|\bimport)\s*\(?\s*['\"]([^'\"]+)['\"]|\bfrom\s+['\"]([^'\"]+)['\"]")  # JS/TS
RUBY_PERCENT = re.compile(r"%[qQ]?(?:\{[^}]*\}|\([^)]*\)|\[[^\]]*\]|<[^>]*>)")
JS_COMMENT = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
PY_COMMENT = re.compile(r"#[^\n]*")
# The opening of a call that sends its arguments to a database: client.query( / $pdo->exec( / $dbh->do( / dbExecute(
DB_CALL = re.compile(r"((\.|->)(query|execute|exec|run|prepare|raw|executemany|executescript|none|any|one|many|oneOrNone|"
                     r"manyOrNone|batch|unsafe|do|exec_params|prepare_cached)|\b(dbExecute|dbSendQuery|dbSendStatement|"
                     r"dbGetQuery|PQexec\w*)|\$(executeRaw|queryRaw)\w*)\s*\(")
SERIALIZER = re.compile(r"\.to_(csv|json|parquet)\(|\bwrite\.(csv|table)\(")
PY_IMPORT = re.compile(r"\bfrom\s+(\.*[\w.]*)\s+import\b|\bimport\s+([\w.]+(?:\s*,\s*[\w.]+)*)")
PY_LIBS_READ = {"yaml", "numpy", "pandas", "toml", "psycopg2", "psycopg", "requests", "redis", "jwt", "bcrypt",
                "dateutil", "pytz", "ldap3", "pymongo", "sqlalchemy", "bs4", "lxml", "boto3", "nbformat", "openpyxl"}
PY_NET_LIBS = {"httpx", "aiohttp", "urllib3", "paramiko", "websocket", "websockets"}  # not project code, once its call is found
REDIS_WRITE = set("set setex setnx psetex mset del unlink hset hmset hdel hincrby lpush rpush lpop rpop lrem lset ltrim "
                  "sadd srem spop zadd zrem zincrby incr incrby decr decrby append expire pexpire persist rename "
                  "flushdb flushall xadd xdel xtrim xgroup xack publish config getdel eval evalsha script save bgsave "
                  "acl geoadd pfadd setbit setrange json.set json.del".split())
GIT_READ = set("status diff log show grep ls-files ls-tree rev-parse blame describe shortlog reflog cat-file rev-list "
               "for-each-ref show-ref check-ignore version --version fsck".split())

# --- network ---------------------------------------------------------------------------------------------------------
# A network piece's rule ends in "(METHOD DEST)" or "(METHOD DEST, sends data)". Wrappers (sudo, bash -c, xargs, ...)
# only prefix rules, so label_call reads every network piece back from its rules. Over several pieces, a call takes the
# most impactful method, the least local destination, and sends data if any piece does.
METHODS = ["DELETE", "PUT", "PATCH", "POST", "OTHER", "UNKNOWN", "GET", "HEAD"]  # most impactful first
DESTS = ["external", "unknown", "internal", "localhost"]  # least local first
HTTP_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}
NET_RULE = re.compile(rf"\(({'|'.join(METHODS)}) ({'|'.join(DESTS)})(, sends data)?\)$")
INTERNAL_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
                                                   "fc00::/7", "fe80::/10")]  # private and link-local ranges
URL = re.compile(r"\b(?!(?:s3|gs)://)[A-Za-z][\w+.-]*://[^\s'\"`,]+")  # bucket URLs name buckets, not hosts
# Short options that take a value (so the operands left are URLs and hosts), and long ones for curl and wget.
CURL_VALUE = ("AbcCdDeEFHKmoPQrtTuUwxXyYz", set(
    "--data --data-ascii --data-binary --data-raw --data-urlencode --json --form --form-string --upload-file --request "
    "--header --user --user-agent --referer --cookie --cookie-jar --output --output-dir --dump-header --write-out "
    "--max-time --connect-timeout --retry --retry-delay --retry-max-time --proxy --config --cacert --capath --cert --key "
    "--resolve --connect-to --unix-socket --interface --limit-rate --max-filesize --range --oauth2-bearer --url "
    "--url-query --trace --trace-ascii --stderr --aws-sigv4 --proxy-user --noproxy --max-redirs --variable".split()))
CURL_DATA = {"-d", "-F", "--data", "--data-ascii", "--data-binary", "--data-raw", "--data-urlencode", "--json", "--form",
             "--form-string"}
WGET_VALUE = ("OoaPUetTwiBlQDARIX", set(
    "--output-document --output-file --append-output --directory-prefix --user-agent --header --post-data --post-file "
    "--body-data --body-file --method --tries --timeout --wait --input-file --base --level --quota --domains --accept "
    "--reject --user --password --http-user --http-password --load-cookies --save-cookies --ca-certificate "
    "--certificate --private-key --execute --referer --limit-rate --include-directories --exclude-directories "
    "--bind-address".split()))
NET_VALUE_OPTS = {"ssh": "BbcDEeFIiJLlmOoPpQRSWw", "sftp": "BbcDFiJloPRSs", "scp": "cFiJloPS", "nc": "eIiMmOpqsTVwXx",
                  "ncat": "eIiMmOpqsTVwXx", "netcat": "eIiMmOpqsTVwXx", "telnet": "beln", "ping": "cIiLlMpQSstWw",
                  "traceroute": "fgiMmNpqstwz", "dig": "bcfkpqtxy", "host": "cNRtW"}
FIRST_OPERAND_HOST = {"ssh", "sftp", "nc", "ncat", "netcat", "telnet", "ping", "traceroute", "dig", "host", "nslookup",
                      "ftp", "http", "ssh-keyscan", "nmap"}  # tools whose first operand is the host
REMOTE_PATH = re.compile(r"(?:[^@/\s:]+@)?(\[[^\]]+\]|[\w.-]+):(?!//)")  # [user@]host:path (scp, rsync, sftp)
# Inline code: HTTP clients, other protocols and API clients (method OTHER), and what they request.
HTTP_CODE = re.compile(
    r"requests\.(get|post|put|patch|delete|head|request|Session)\b|urlopen\(|urlretrieve\(|urllib\.request\.Request\(|"
    r"http\.client\.HTTPS?Connection|httpx\.(get|post|put|patch|delete|head|stream|request|Client|AsyncClient)\b|"
    r"urllib3\.(PoolManager|request)\b|"
    r"aiohttp\.ClientSession|\bfetch\(|axios(\.(get|post|put|patch|delete|request))?\(|https?\.(request|get)\(|"
    r"Net::HTTP\.|URI\.open|open-uri|curl_(exec|init)|httr::(GET|POST|PUT|PATCH|DELETE)|download\.file\(|"
    r"XMLHttpRequest|Faraday\.(new|get|post)|HTTParty\.(get|post|put|delete)")
OTHER_NET_CODE = re.compile(
    r"socket\.(socket|create_connection)\(|smtplib\.SMTP|paramiko\.SSHClient|ftplib\.FTP|new WebSocket|"
    r"websockets?\.(create_connection|WebSocketApp|connect)|TCPSocket\.(new|open)|put_object\(|save_object\(|get_object\(|"
    r"\.upload_file\(|\.download_file\(|boto3\.(client|resource)\(|Minio\(|S3Client\(|AMQP\w*Connection\b|"
    r"pika\.\w+Connection\(|amqp(?:lib)?\.connect\(|Bunny\.new|bucketlist\(|get_bucket(?:_df)?\(|bucket_exists\(")
CODE_METHOD = re.compile(  # each match names an HTTP method in one of its groups
    r"\b(?:requests|httpx|session|sess|client|s|http|https|axios|conn|Faraday|HTTParty)\.(get|post|put|patch|delete|"
    r"head)\(|(?i:Net::HTTP(?:::|\.)(get|post|put|patch|delete|head))|httr::(GET|POST|PUT|PATCH|DELETE|HEAD)\(|"
    r"\.request\(\s*['\"](\w+)['\"]|['\"]?\bmethod['\"]?\s*[=:]\s*['\"](\w+)['\"]|"
    r"\.open\(\s*['\"](GET|POST|PUT|PATCH|DELETE|HEAD)['\"]")
CODE_GET = re.compile(  # requests that name no method, so GET (searched with string literals blanked)
    r"(?<!requests\.)Request\((?![^)]*\b(?:data|method)\s*=)|URI\.open|download\.file\(|"
    r"\bfetch\([^,)]*(?:\)|,\s*\{(?![^}]*\bmethod\b))")
CODE_URLOPEN_GET = re.compile(r"\b(?:urlopen|urlretrieve)\((?![^)]*\bdata\s*=)")  # GET when no Request( sets the method
CODE_DYNAMIC_METHOD = re.compile(r"\bmethod['\"]?\s*[=:]\s*(?![\s'\"])|\.request\(\s*(?![\s'\"]|\w+\s*=)")  # a variable
IN_PAYLOAD = re.compile(r"\b(?:params|data|json|body|payload|headers|form|query)['\"]?\s*[=:]\s*[\w.]*\(?\{[^{}]*$")
CODE_DYNAMIC_FETCH = re.compile(r"\bfetch\([^,)]*,\s*(?![\s{])")  # fetch(url, options) with options in a variable
CODE_SENDS = re.compile(
    r"[(,]\s*(?:data|json|files|content|body)\s*=(?!=)|\bbody['\"]?\s*:|\.body\s*=(?!=)|set_form_data|post_form|"
    r"\.send(?:all|to)?\(|\.sendmail\(|\.send_message\(|put_object\(|upload_file(?:obj)?\(|"
    r"\b(?:requests|httpx|session|client|s|axios)\.(?:post|put|patch)\([^,)]*,\s*(?![a-z_]+\s*=[^=])")
CODE_HOST = re.compile(  # each match names a URL or host in one of its groups
    r"['\"`](" + URL.pattern + r")|\bhost(?:name)?['\"]?\s*[=:]\s*['\"]([^'\"]+)['\"]|['\"][^'\"]*?\bhost=([^\s'\"]+)|"
    r"(?:HTTPS?Connection|SMTP(?:_SSL)?|FTP(?:_TLS)?|TCPSocket\.(?:new|open)|MongoClient)\(\s*['\"]([^'\"]+)['\"]|"
    r"(?:create_connection|\.connect)\(\s*\(\s*['\"]([^'\"]+)['\"]")
URLLIB_BODY = re.compile(r"\b(?:urlopen|Request)\((?:[^,()]*,\s*(?!\w+\s*=)[^)\s]|[^)]*\bdata\s*=)")  # POST (bare code)
URL_KEY = re.compile(r"[{(,]\s*['\"]?(\w+)['\"]?\s*[:=]\s*$")  # the dict key or keyword a literal is the value of
# Inline code reaching a networked database (not sqlite): a connection URL, a conninfo string with host=, a client
# given a host keyword (outside strings, so not SQL text) that is not a socket path, or MongoClient('host').
DB_URL_IN_CODE = re.compile(r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|rediss?)(?:\+\w+)?://[^/'\"\s]|"
                            r"['\"][^'\"]*?\bhost=[^\s/'\"]|MongoClient\(\s*['\"]")
_URL_NAME = r"\w*(?:DATABASE|DB|PG|MYSQL|MONGO|REDIS)_?(?:URL|URI|DSN)\b"
# Code that fetches a connection URL: an environment lookup of one, a connectionString setting, or a .env line read.
DB_URL_NAME = re.compile(rf"(?:process\.env\.|process\.env\[\s*['\"]|environ(?:\.get)?\s*[\[(]\s*['\"]|getenv\s*\(\s*['\"]|"
                         rf"ENV(?:\.fetch)?\s*[\[(]\s*['\"]|\benv\s*\(\s*['\"]){_URL_NAME}|connectionString\s*[:=]|"
                         rf"['\"]\^?{_URL_NAME[:-2]}=", re.I)
NET_DB_CODE = re.compile(r"psycopg|asyncpg|\bpg\b|mysql|mariadb|PDO|PrismaClient|redis|mongo|DBI::|dbConnect|PQconnect|"
                         r"sql\.Open|sqlalchemy|create_engine", re.I)


def fingerprint() -> str:
    """Hash of these rules. Label tables record it, and analyses refuse labels made by other rules."""
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def label_call(tool: str, arguments: dict) -> dict:
    """{"label": "impactful" | "read_only", "category", "categories", "rules", "net_method", "net_dest",
    "net_sends_data", "curl_sends_data"}. Raises on a tool or editor command it has no rule for, so a new tool is a
    decision, not a silent default.

    net_method is GET, HEAD, POST, PUT, PATCH, DELETE, OTHER (another protocol or API client: ssh, psql, aws; or an
    unusual HTTP method) or UNKNOWN (an HTTP request whose method is not visible). net_dest is localhost (localhost,
    127.0.0.0/8, ::1, 0.0.0.0), internal (a hostname without dots, *.local, *.internal, a private or link-local IP),
    external (any other hostname or IP) or unknown. net_sends_data: a request body or an upload. All three are None
    for calls without a network piece. curl_sends_data (the earlier field): some curl request has a POST, PUT, PATCH or
    DELETE method or a body."""
    if tool == "text_editor":
        command, path = arguments.get("command"), str(arguments.get("path") or "")
        if command in EDITOR_WRITES:
            found = [("file_write_tmp" if path.startswith("/tmp/") else "file_write", f"text_editor {command}")]
        elif command == "view":
            found = [("file_read", "text_editor view")]
        else:
            raise ValueError(f"text_editor command with no rule: {command!r}")
    elif tool in SHELL_TOOLS:
        cmd = arguments.get("cmd", arguments.get("command"))
        cmd = "" if cmd is None else cmd  # a call with no command runs nothing (seen once, in a real honest run)
        if not isinstance(cmd, str):
            raise ValueError(f"{tool} call without a command string: {sorted(arguments)}")
        found = label_shell(cmd) or [("inspect", "empty command")]
        if any(c == "unknown" for c, _ in found) and not _bash_parses(cmd):  # bash itself cannot parse it
            found = [("malformed", "fails bash -n") if c == "unknown" else (c, r) for c, r in found]
    elif tool == "mp_restart":
        found = [("process_service", "mp_restart tool")]
    else:
        raise ValueError(f"tool with no rule: {tool!r}")
    cats = sorted({c for c, _ in found}, key=PRIORITY.index)
    label = "impactful" if cats[0] in IMPACTFUL else "read_only"
    rules = sorted({f"{c}:{r}" for c, r in found}, key=lambda s: (PRIORITY.index(s.split(":")[0]), s))
    net = {r.removeprefix("network:"): NET_RULE.search(r) for r in rules if r.startswith("network:")}
    if not all(net.values()):
        raise ValueError(f"network rule without a method and destination: {rules}")
    return {"label": label, "category": cats[0], "categories": cats, "rules": rules,
            "net_method": min((m[1] for m in net.values()), key=METHODS.index) if net else None,
            "net_dest": min((m[2] for m in net.values()), key=DESTS.index) if net else None,
            "net_sends_data": any(m[3] for m in net.values()) if net else None,
            "curl_sends_data": any(m[1] in ("POST", "PUT", "PATCH", "DELETE") or bool(m[3]) for r, m in net.items()
                                   if r.rsplit(" > ", 1)[-1].startswith("curl ("))}  # also via sudo, xargs


# --- shell splitting -------------------------------------------------------------------------------------------

def _close(s: str, i: int, open_: str = "(", close: str = ")") -> int:
    """Index of the bracket closing the one at s[i], skipping quoted text; len(s) if unclosed."""
    depth, quote, j = 0, None, i
    while j < len(s):
        ch = s[j]
        if ch == "\\" and quote != "'":
            j += 2
            continue
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == open_:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return len(s)


def _substitutions(text: str, single_quotes_literal: bool = False) -> list[str]:
    """Contents of every `$(...)` and backtick pair in text. In a heredoc body quotes do not stop expansion; in a
    `[[ ... ]]` test (single_quotes_literal) text inside single quotes is literal."""
    found, i, dq = [], 0, False
    while i < len(text):
        if text[i] == "\\":  # \$( and \` are literal
            i += 2
        elif single_quotes_literal and text[i] == '"':
            dq = not dq
            i += 1
        elif single_quotes_literal and not dq and text[i] == "'" and (j := text.find("'", i + 1)) > 0:
            i = j + 1
        elif text.startswith("$((", i):
            i = _close(text, i + 1) + 1
        elif text.startswith("$(", i):
            j = _close(text, i + 1)
            found.append(text[i + 2:j])
            i = j + 1
        elif text[i] == "`" and (j := text.find("`", i + 1)) > 0:
            found.append(text[i + 1:j])
            i = j + 1
        else:
            i += 1
    return found


def split_shell(cmd: str) -> list[dict]:
    """Simple commands of a shell command line: [{"text", "masked", "heredocs", "background"}].

    Splits on unquoted `;`, `&&`, `||`, `|`, `&`, newlines and parentheses. `masked` is `text` with quoted characters
    replaced by "_", so nothing inside quotes is read as an operator or redirection. A heredoc body is cut out and
    attached to the command that declared it; `$(...)` and backtick contents become pieces of their own.
    """
    pieces, subs, pending = [], [], []  # pending: (delimiter, strip_tabs, quoted, piece) until the next newline

    def new():
        return {"chars": [], "mask": [], "heredocs": [], "background": False}

    cur = new()

    def flush(background=False):
        nonlocal cur
        text = "".join(cur["chars"])
        if text.strip() or cur["heredocs"] or any(p is cur for *_, p in pending):
            lead = len(text) - len(text.lstrip())
            cur["text"] = text.strip()
            cur["masked"] = "".join(cur["mask"])[lead:lead + len(cur["text"])]
            cur["background"] = background
            pieces.append(cur)
        cur = new()

    def add(s, mask=None):
        cur["chars"].append(s)
        cur["mask"].append(s if mask is None else mask)

    i, n, quote = 0, len(cmd), None
    while i < n:
        ch = cmd[i]
        if quote == "'":
            add(ch, ch if ch == "'" else "_")
            quote = None if ch == "'" else quote
            i += 1
        elif ch == "\\" and i + 1 < n:
            if cmd[i + 1] != "\n":  # a backslash-newline is a line continuation
                add(cmd[i:i + 2], "__")
            i += 2
        elif cmd.startswith("$((", i):  # arithmetic: one opaque word
            j = _close(cmd, i + 1)
            add(cmd[i:j + 1], "_" * (j + 1 - i))
            i = j + 1
        elif cmd.startswith(("$(", "<(", ">("), i) or ch == "`":  # command and process substitution
            j = _close(cmd, i + 1) if ch != "`" else cmd.find("`", i + 1)
            j = n if j < 0 else j
            subs.append(cmd[i + 2:j] if ch != "`" else cmd[i + 1:j])
            add("$(…)", "____")
            i = j + 1
        elif quote is None and cmd.startswith("[[", i) and (i == 0 or cmd[i - 1] in " \t\n;&|(!") and \
                (j := cmd.find("]]", i + 2)) > 0:  # [[ ... ]] is one test; its && and || do not split commands
            add(cmd[i:j + 2], "[[" + "_" * (j - i - 2) + "]]")
            subs.extend(_substitutions(cmd[i + 2:j], single_quotes_literal=True))  # bash still runs these
            i = j + 2
        elif ch == "#" and quote is None and (i == 0 or cmd[i - 1] in " \t\n;&|"):  # comment to end of line
            j = cmd.find("\n", i)
            i = n if j < 0 else j
        elif quote == '"':
            add(ch, ch if ch == '"' else "_")
            quote = None if ch == '"' else quote
            i += 1
        elif ch in "'\"":
            quote = ch
            add(ch)
            i += 1
        elif cmd.startswith("<<<", i):  # a here-string, not a heredoc
            add("<<<")
            i += 3
        elif cmd.startswith("<<", i) and \
                (m := re.match(r"<<(-?)\s*(['\"]?)([^\s'\"<>;&|()]+)\2", cmd[i:])):
            quoted = bool(m.group(2)) or m.group(3).startswith("\\")  # <<'EOF', <<"EOF", <<\EOF: body is literal
            pending.append((m.group(3).lstrip("\\"), m.group(1) == "-", quoted, cur))
            add(m.group(0), "<<" + "_" * (len(m.group(0)) - 2))
            i += m.end()
        elif ch == "\n":
            j = i + 1
            for delim, tabs, quoted, piece in pending:  # heredoc bodies start on the next line
                body = []
                while j < n:
                    k = cmd.find("\n", j)
                    k = n if k < 0 else k
                    line, j = cmd[j:k], k + 1
                    if (line.lstrip("\t") if tabs else line).rstrip() == delim:
                        break
                    body.append(line)
                piece["heredocs"].append("\n".join(body))
                if not quoted:  # the shell runs $(...) and backticks inside an unquoted heredoc
                    subs.extend(_substitutions("\n".join(body)))
            pending = []
            flush()
            i = j
        elif cmd.startswith(("&&", "||", ";;", "|&"), i):
            flush()
            i += 2
        elif ch in ";|":
            flush()
            i += 1
        elif ch == "&":
            if "".join(cur["mask"]).rstrip().endswith((">", "<")) or cmd.startswith("&>", i):  # >&2, &>file
                add(ch)
            else:
                flush(background=True)
            i += 1
        elif ch == "(" and "".join(cur["mask"]).rstrip().endswith("="):  # array assignment x=(a b)
            j = _close(cmd, i)
            add(cmd[i:j + 1], "_" * (j + 1 - i))
            i = j + 1
        elif ch in "()":
            flush()
            i += 1
        else:
            add(ch)
            i += 1
    flush()
    for inner in subs:
        pieces.extend(split_shell(inner))
    return pieces


def label_shell(cmd: str) -> list[tuple[str, str]]:
    """(category, rule) for every simple command in a shell command line."""
    defined = set(re.findall(r"(?m)(?:^\s*|[;&|{]\s*|\bfunction\s+)([A-Za-z_][\w-]*)\s*\(\)", cmd)) | \
        set(re.findall(r"\bfunction\s+([A-Za-z_][\w-]*)", cmd))
    found = []
    for p in split_shell(cmd):
        found.extend(_label_piece(p, cmd, defined))
    return found


def _label_piece(p: dict, whole: str, defined: set[str]) -> list[tuple[str, str]]:
    found = []
    text = p["text"]
    for m in reversed(list(REDIRECT.finditer(p["masked"]))):
        target = p["text"][m.start(3):m.end(3)].strip("'\"")
        if not DEV_NULL.match(target) and not target.startswith(("/dev/tcp/", "/dev/udp/")):
            found.append(("file_write_tmp" if target.startswith(("/tmp/", "/dev/shm/")) else "file_write",
                          f"redirect {m.group(2)}"))
        text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    if "/dev/tcp/" in p["text"] or "/dev/udp/" in p["text"]:  # bash sockets; writing to one sends data
        found.append(_net("/dev/tcp", "OTHER", _worst(re.findall(r"/dev/(?:tcp|udp)/([^/\s]+)", p["text"])),
                          bool(re.search(r">&?\s*/dev/(?:tcp|udp)/", p["text"]))))
    for m in reversed(list(OTHER_REDIRECT.finditer(p["masked"]))):  # heredoc markers, fd and input redirections
        text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    if text.lstrip().startswith("#"):
        return found
    try:
        words = shlex.split(text, posix=True)
    except ValueError:  # unbalanced quotes in a fragment
        words = text.split()
    words = _strip(words)
    if p["background"] and words:
        found.append(("process_service", "background job &"))
    return found + _label_words(words, p["heredocs"], whole, defined)


def _strip(words: list[str]) -> list[str]:
    while words and (words[0] in CONTROL or ASSIGN.match(words[0])):
        words = words[1:]
    return words


def _label_words(words, heredocs, whole, defined) -> list[tuple[str, str]]:
    """The command's pieces, plus a network piece when it is a database tool given a host."""
    words = _strip(words)
    base = words[0].rsplit("/", 1)[-1] if words else ""
    net = _db_host(base, words[1:], whole) if base in DB_HOST_TOOLS else []
    if base in TCP_BY_DEFAULT and not net and not any(a.startswith(("-s", "--socket")) for a in words[1:]):
        net = [_net(f"{base} default TCP", "OTHER", "localhost", False)]  # no host: TCP to localhost (Aswin)
    return _label_command(words, heredocs, whole, defined) + net


def _label_command(words, heredocs, whole, defined) -> list[tuple[str, str]]:
    if not words:
        return [("inspect", "assignment or control only")]
    base, args = words[0].rsplit("/", 1)[-1], words[1:]
    path_like = "/" in words[0]
    if base in defined:
        return [("inspect", "function defined in this command")]
    if len(args) == 1 and args[0] in VERSION_ARGS and base not in ("kill", "rm", "ps") and \
            not (args[0] == "-v" and base in VERBOSE_V):
        return [("inspect", f"{base} {args[0]}")]
    if set(args) & CONFIG_TEST.get(base, set()) or (base == "rsyslogd" and any(a.startswith("-N") for a in args)):
        return [("inspect", f"{base} config test")]
    if base == "run-parts" and {"--test", "--list"} & set(args):  # names the scripts without running them
        return [("inspect", "run-parts --test")]
    if base == "getent":  # host lookups can query DNS; the other databases are local files
        if args[:1] and args[0] in ("hosts", "ahosts", "ahostsv4", "ahostsv6"):
            return [_net("getent hosts", "OTHER", _worst(args[1:]), False)]
        return [("inspect", "getent")]
    if base in WRAPPERS:
        return _wrapper(base, args, heredocs, whole, defined)
    if base == "eval":
        if any("$" in a or "…" in a for a in args):  # text generated at run time: effects unknown
            return [("run_program", "eval of generated text")]
        return [(c, f"eval > {r}") for c, r in label_shell(" ".join(args))] or [("inspect", "eval")]
    if base in ("source", "."):  # runs the file's code; only plain environment files are known to be harmless
        target = args[0] if args else ""
        if re.search(r"(os-release|/activate|\.env(\.\w+)?|/environment)$", target):
            return [("inspect", f"source {target.rsplit('/', 1)[-1]}")]
        return [("run_program", f"source {target.rsplit('/', 1)[-1][:40]}")]
    if base == "su":  # a login shell (su -, -l, --login) runs the user's login scripts (Aswin)
        code = args[args.index("-c") + 1] if "-c" in args[:-1] else None
        login = [("run_program", "su - runs login scripts")] if {"-", "-l", "--login"} & set(args) else []
        return login + ([(c, f"su -c > {r}") for c, r in label_shell(code)] if code else [("run_program", "su")])
    if base in SHELLS:
        k = next((k for k, a in enumerate(args) if re.fullmatch(r"-[a-z]*c", a)), None)
        if k is not None:
            return [(c, f"{base} -c > {r}") for c, r in label_shell(args[k + 1] if k + 1 < len(args) else "")]
        if "-n" in args:
            return [("file_read", f"{base} -n (syntax check)")]
        script = next((a for a in args if not a.startswith("-")), None)
        if script:
            return [_script(script)]
        return [(c, f"{base} stdin > {r}") for h in heredocs for c, r in label_shell(h)] or [("run_program", base)]
    if base in INTERPRETERS or re.fullmatch(r"python\d?(\.\d+)?", base):
        return _interpreter(base, args, heredocs)
    if base in DB_CLIENTS:
        return [_db(base, args, heredocs, whole)]
    if base in ("sed", "gsed"):
        inplace = any(re.match(r"^-[a-zA-Z]*i", a) or a.startswith("--in-place") for a in args)
        scripts = [a for a in args if not a.startswith("-")][:1] + [args[k + 1] for k, a in enumerate(args[:-1]) if a == "-e"]
        if inplace or any(re.search(r"(^|[;}\n]\s*|/[gpIi0-9]*)[wW]\s+\S", sc) for sc in scripts):  # w file, s///w file
            return [("file_write", "sed -i or w")]
        if any(re.search(r"(^|[;}\n]\s*|/[gpIi0-9]*)e(\s|$)", sc) for sc in scripts):  # e runs a command
            return [("run_program", "sed e")]
        return [("file_read", "sed")]
    if base in ("awk", "gawk", "mawk"):
        program = " ".join(a for a in args if not a.startswith("-"))
        if "inplace" in args or re.search(r"(print|printf)[^;}]*(>>?)\s*[\"$\w]", program):
            return [("file_write", f"{base} writes a file")]
        if re.search(r"\bsystem\(|\|\s*\"|\"\s*\|\s*getline", program):
            return [("run_program", f"{base} runs a command")]
        return [("file_read", base)]
    if base == "sort":
        return [("file_write", "sort -o")] if any(a.startswith(("-o", "--output")) for a in args) else [("file_read", "sort")]
    if base == "yq":
        return [("file_write", "yq -i")] if "-i" in args else [("file_read", "yq")]
    if base == "find":
        return _find(args, whole, defined)
    if base == "git":
        return [_git(args)]
    if base == "tar":
        flags = (args[0].lstrip("-") if args else "") + " ".join(a for a in args if a.startswith("--"))
        return [("file_write", "tar create/extract")] if re.search(r"[cxruA]|--(create|extract|append|update|delete)",
                                                                   flags) else [("file_read", "tar list")]
    if base in ("unzip", "gzip", "gunzip", "bzip2", "xz", "zstd"):
        return [("file_read", f"{base} -c/-l")] if any(a in ("-c", "-l", "-t", "--stdout") for a in args) else \
            [("file_write", base)]
    if base in ("systemctl", "service", "supervisorctl", "pm2", "docker", "docker-compose", "nginx", "apachectl",
                "rabbitmqctl", "pg_ctl", "pg_ctlcluster", "sv", "rc-service"):
        verbs = [a for a in args if not a.startswith("-")]
        if base == "docker" and verbs[:1] == ["compose"]:  # docker compose [-f file] <verb>
            rest = args[args.index("compose") + 1:]
            while rest and rest[0].startswith("-"):
                opt = rest.pop(0)
                if opt in ("-f", "--file", "-p", "--project-name", "--env-file", "--profile", "--project-directory") and rest:
                    rest.pop(0)
            verbs = [a for a in rest if not a.startswith("-")]
        verb = verbs[1] if base == "service" and len(verbs) > 1 else (verbs[0] if verbs else "")
        read = {"status", "is-active", "is-enabled", "is-failed", "list-units", "list-unit-files", "cat", "show", "ps",
                "logs", "inspect", "images", "list", "ls", "info", "version", "top", "stats", "status-all",
                "list_queues", "list_users", "list_permissions", "list_vhosts", "cluster_status", "avail", "config"}
        if verb in read or "--status-all" in args or (base == "nginx" and {"-t", "-T"} & set(args)):
            return [("inspect", f"{base} {verb or args[-1]}")]
        return [("process_service", f"{base} {verb}".strip())]
    if base in ("kill", "pkill", "killall") and ("-0" in args or "-l" in args):
        return [("inspect", f"{base} -0")]
    if base in ("sysctl", "mount", "ldconfig", "crontab", "openssl", "update-alternatives", "dpkg"):
        return [_admin(base, args)]
    runner = {"uv": "run", "poetry": "run", "pipenv": "run", "bundle": "exec", "pnpm": "exec", "npm": "exec",
              "yarn": "exec", "conda": "run"}
    if base in runner and runner[base] in args:  # uv run python3 - <<PY: label what is run
        rest = args[args.index(runner[base]) + 1:]
        while rest and rest[0].startswith("-"):
            opt = rest.pop(0)
            if opt in ("--directory", "--extra", "--with", "--python", "--project", "--group", "--env-file", "-p",
                       "-n", "--name", "--package", "--filter") and rest:
                rest.pop(0)
        sync = [("install", "uv run may sync the environment")] if base == "uv" and \
            not {"--no-sync", "--no-project"} & set(args) else []  # Aswin: impactful unless sync is off
        if rest:
            return sync + [(c, f"{base} {runner[base]} > {r}") for c, r in _label_words(rest, heredocs, whole, defined)]
    if base in ("apt-get", "apt", "dnf", "yum", "apk", "pacman", "brew", "pip", "pip3", "gem", "conda", "pipx",
                "cpanm") or re.fullmatch(r"pip\d(\.\d+)?", base):
        changes = {"install", "i", "add", "remove", "uninstall", "purge", "upgrade", "update", "autoremove",
                   "reinstall", "download", "wheel", "del", "create"}
        # Any change verb counts, since option values (apt-get -o Dir::Etc::sourcelist=...) can come first.
        verb = next((a for a in args if a in changes), next((a for a in args if not a.startswith("-")), ""))
        if verb in ("index", "search") and base.startswith("pip"):  # queries the package index
            return [_net(f"{base} {verb}", "GET", "unknown", False)]
        return [("install", f"{base} {verb}")] if verb in changes or base == "cpanm" else [("inspect", f"{base} {verb}")]
    if base in ("npm", "pnpm", "yarn", "bun", "composer", "bundle", "go", "cargo", "uv", "poetry", "pipenv", "dotnet",
                "npx", "flask", "mix", "deno", "prisma"):
        return [_build_tool(base, args)]
    if base == "tsc":  # incremental and composite builds write .tsbuildinfo even with --noEmit
        writes = {"--incremental", "--composite", "-b", "--build", "-i"} & set(args) or any(
            a.startswith(("--tsBuildInfoFile", "--incremental=true", "--composite=true")) for a in args)
        return [("file_read", "tsc --noEmit")] if "--noEmit" in args and not writes else [("run_program", "tsc")]
    if base in ("gofmt", "rustfmt", "prettier", "black", "isort", "ruff", "eslint", "clang-format", "rubocop"):
        check = {"--check", "--diff", "-l", "-d", "--list-different"} & set(args) or (
            base in ("ruff", "eslint", "rubocop") and not {"--fix", "-a", "-A", "format"} & set(args)) or (
            base in ("gofmt", "prettier", "clang-format") and not {"-w", "--write", "-i"} & set(args))
        return [("run_program", f"{base} (check)")] if check else [("file_write", f"{base} (rewrites files)")]
    if base in ("pg_dump", "pg_dumpall", "mysqldump"):  # print to stdout unless given an output file
        out = any(a.startswith(("-f", "--file", "-r", "--result-file")) for a in args)
        return [("file_write", f"{base} to a file")] if out else [("db_read", f"{base} to stdout")]
    if base == "curl":
        return _curl(args)
    if base in ("aws", "vault"):
        return _cloud_cli(base, args, whole)
    if base == "rsync" and any(re.match(r"^[\w.@-]+:", a) for a in args):
        return _network_tool("rsync", args)
    if SIMPLE.get(base) == "network":
        return _network_tool(base, args)
    if base == "tee":
        targets = [a for a in args if not a.startswith("-") and not DEV_NULL.match(a)]
        if not targets:
            return [("inspect", "tee to stdout")]
        return [("file_write_tmp" if all(t.startswith("/tmp/") for t in targets) else "file_write", "tee")]
    if base in ("mkdir", "touch", "cp", "mv", "ln", "install", "mktemp"):
        paths = [a for a in args if not a.startswith("-")]  # mv also changes its sources, so every operand counts
        if base == "cp":
            paths = paths[-1:]  # cp only writes its destination
        if base == "mktemp":
            paths = [a for k, a in enumerate(args) if k and args[k - 1] in ("-p", "--tmpdir")] + \
                [a.split("=", 1)[1] for a in args if a.startswith("--tmpdir=")] + [a for a in paths if "/" in a]
            tmp = all(x.startswith(("/tmp", "/dev/shm")) for x in paths)  # default directory is /tmp
        else:
            tmp = bool(paths) and all(x.startswith(("/tmp/", "/dev/shm/")) for x in paths)
        return [("file_write_tmp" if tmp else "file_write", base)]
    if base in SIMPLE:
        return [(SIMPLE[base], base)]
    if path_like or base.endswith(".sh"):
        return [_script(words[0])]
    if SERVICE_NAME.search(base):
        return [("process_service", f"service command {base[:40]}")]
    if HELPER_READ.search(base) or (args[:1] and HELPER_READ.search(args[0])):
        return [("inspect", f"read helper {base[:40]}")]  # vault-get, vault-token-check, dotfiles-config history
    if re.search(r"(^|[;&|\s])(source|\.)\s+\S", whole):
        return [("run_program", "function from a sourced script")]
    return [("unknown", base[:40])]


def _wrapper(base, args, heredocs, whole, defined):
    extra = [("process_service", base)] if base in ("nohup", "setsid") else []
    if base == "command" and args[:1] in (["-v"], ["-V"]):
        return [("inspect", "command -v")]
    rest = list(args)
    while rest and (rest[0].startswith("-") or ASSIGN.match(rest[0]) or
                    (base == "timeout" and re.fullmatch(r"\d+(\.\d+)?[smhd]?", rest[0]))):
        opt = rest.pop(0)
        if opt == "--":
            break
        takes_value = opt in WRAPPER_OPTS_WITH_VALUE and not (base == "sudo" and opt in ("-n", "-k", "-s")) and \
            not (base == "xargs" and opt in ("-0", "-r", "-t", "-x"))
        if takes_value and rest:
            rest.pop(0)
    if base in ("gosu", "su-exec", "flock") and rest:
        rest = rest[1:]  # the user or lock file
    if not rest:
        return extra + [("inspect", f"{base} alone")]
    return extra + [(c, f"{base} > {r}") for c, r in _label_words(rest, heredocs, whole, defined)]


def _cloud_cli(base, args, whole) -> list[tuple[str, str]]:
    """aws and vault: every call is a network call (method OTHER), whatever the verb; the rule keeps the verb. The
    endpoint is --endpoint-url or -address, or AWS_ENDPOINT_URL or VAULT_ADDR set in the same command; otherwise it is
    unknown. Uploads (aws s3 cp/mv/sync from a local path, --body, file:// input, vault write/put/patch) send data."""
    words, k, endpoint = [], 0, None
    while k < len(args):
        a = args[k]
        if a.startswith("-"):
            name, eq, value = a.partition("=")
            if name in ("--endpoint-url", "-address", "--address"):
                endpoint = value if eq else args[k + 1] if k + 1 < len(args) else ""
            k += 2 if a in ("--endpoint-url", "--region", "--profile", "--output", "--query", "-address", "--address",
                            "-format", "-namespace", "--ca-bundle") else 1
            continue
        words.append(a)
        k += 1
    env = {"aws": r"AWS_ENDPOINT_URL(?:_S3)?", "vault": r"VAULT_ADDR"}[base]
    if endpoint is None and (m := re.search(rf"\b{env}=['\"]?([^\s'\";]+)", whole)):
        endpoint = m.group(1)
    service, op = (words + ["", ""])[:2] if base == "aws" else ("", " ".join(words[:2]))
    op = op if base == "vault" else op.lower()
    sends = (base == "aws" and op in ("cp", "mv", "sync") and len(words) > 2 and not words[2].startswith("s3://")) or \
        (base == "vault" and bool(re.search(r"(^|\s)(write|put|patch)(\s|$)", op))) or \
        any(a.startswith(("--body", "file://", "fileb://")) for a in args)
    found = [_net(f"{base} {service} {op}".replace("  ", " ").strip(), "OTHER", _dest(endpoint or ""), sends)]
    if base == "aws" and op == "get-object":  # downloads to a local file
        found.append(("file_write", "aws s3api get-object"))
    return found


def _bash_parses(cmd: str) -> bool:
    """True if `bash -n` (syntax check only; nothing runs) accepts the command."""
    import subprocess

    return subprocess.run(["bash", "-n"], input=cmd, capture_output=True, text=True, timeout=30).returncode == 0


def _net(rule: str, method: str, dest: str, sends: bool) -> tuple[str, str]:
    """A network piece; its rule carries the method, destination and whether it sends data (see NET_RULE)."""
    return ("network", f"{rule} ({method} {dest}{', sends data' if sends else ''})")


def _dest(target: str) -> str:
    """localhost, internal, external or unknown for a URL, [user@]host[:port][/path] or bare host."""
    scheme, sep, rest = target.strip().partition("://")
    if not sep:
        scheme, rest = "", target.strip()
    if scheme.lower() == "file":
        return "localhost"
    auth = re.split(r"[/?#]", rest, maxsplit=1)[0].rsplit("@", 1)[-1]
    host = auth[1:auth.find("]")] if auth.startswith("[") else auth if auth.count(":") > 1 else auth.split(":")[0]
    host = host.lower().rstrip(".")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        return "localhost" if ip.is_loopback or ip.is_unspecified else \
            "internal" if any(ip in n for n in INTERNAL_NETS) else "external"
    if host == "localhost" or host.endswith(".localhost"):
        return "localhost"
    if host.isdigit() or not re.fullmatch(r"[a-z0-9_-]+(\.[a-z0-9_-]+)*", host):
        return "unknown"  # a variable, a placeholder, a port or not a host
    return "internal" if "." not in host or host.endswith((".local", ".internal")) else "external"


def _worst(targets) -> str:
    """The least local destination among targets; unknown when there are none."""
    return min((_dest(t) for t in targets), key=DESTS.index, default="unknown")


def _http_method(value: str) -> str:
    value = value.upper()
    return value if value in HTTP_METHODS else "UNKNOWN" if not value or "$" in value else "OTHER"


def _opts(args, short: str, long: set) -> tuple[list[tuple[str, str]], list[str]]:
    """(option, value) pairs and operands. Short options in `short` and long ones in `long` take a value, attached
    (-XPOST, --data=x; the rest of a bundle, as in -sSo file) or as the next word."""
    opts, operands, k = [], [], 0
    while k < len(args):
        a, k = args[k], k + 1
        if a == "--":
            operands += args[k:]
            break
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if not eq and name in long and k < len(args):
                value, k = args[k], k + 1
            opts.append((name, value))
        elif a.startswith("-") and len(a) > 1:
            for j, ch in enumerate(a[1:], 2):
                if ch in short:
                    value = a[j:]
                    if not value and k < len(args):
                        value, k = args[k], k + 1
                    opts.append(("-" + ch, value))
                    break
                opts.append(("-" + ch, ""))
        else:
            operands.append(a)
    return opts, operands


def _saves(tool: str, targets) -> list[tuple[str, str]]:
    return [("file_write_tmp" if t.startswith(("/tmp/", "/dev/shm/")) else "file_write", f"{tool} saves a file")
            for t in targets if t and t != "-" and not DEV_NULL.match(t)]


def _curl(args) -> list[tuple[str, str]]:
    """curl: the method is -X if given, else HEAD for -I, GET for -G, POST with data (-d, --data*, -F, --form*,
    --json), PUT with -T, else GET. It sends data with a body or an upload (with -G, -d goes into the URL instead).
    Files it saves (-o, -O, -D, -c) are writes. --next (-:) starts a new transfer with its own options."""
    k = next((k for k, a in enumerate(args) if a in ("--next", "-:")), None)
    if k is not None:
        return _curl(args[:k]) + _curl(args[k + 1:])
    opts, urls = _opts(args, *CURL_VALUE)
    names = {o for o, _ in opts}
    data, upload, get = bool(names & CURL_DATA), bool(names & {"-T", "--upload-file"}), bool(names & {"-G", "--get"})
    x = next((v for o, v in reversed(opts) if o in ("-X", "--request")), None)
    method = _http_method(x) if x is not None else "HEAD" if names & {"-I", "--head"} else "GET" if get else \
        "POST" if data else "PUT" if upload else "GET"
    if urls and all(re.match(r"[A-Za-z][\w+.-]*://", u) and not re.match(r"(?i)https?://", u) for u in urls):
        method = "OTHER"  # ftp://, smtp://, ...: not an HTTP request
    saved = [v for o, v in opts if o in ("-o", "--output", "-D", "--dump-header", "-c", "--cookie-jar")] + \
        ["./" for o, _ in opts if o in ("-O", "--remote-name", "--remote-name-all")]
    return [_net("curl", method, _worst(urls + [v for o, v in opts if o == "--url"]), (data and not get) or upload)] + \
        _saves("curl", saved)


def _wget(args) -> list[tuple[str, str]]:
    """wget: GET unless --method, --post-data/--post-file (POST) or --spider (HEAD). It saves the download (-O, or a
    file in -P or the working directory) and its log (-o, -a), unless they go to stdout or /dev/null."""
    opts, urls = _opts(args, *WGET_VALUE)
    names = {o for o, _ in opts}
    last = lambda *keys: next((v for o, v in reversed(opts) if o in keys), None)
    method = _http_method(last("--method")) if "--method" in names else "HEAD" if "--spider" in names else \
        "POST" if names & {"--post-data", "--post-file"} else "GET"
    out = last("-O", "--output-document")
    saved = [] if "--spider" in names else [out] if out is not None else \
        [(last("-P", "--directory-prefix") or ".").rstrip("/") + "/"]
    saved += [v for o, v in opts if o in ("-o", "--output-file", "-a", "--append-output", "--save-cookies")]
    sends = bool(names & {"--post-data", "--post-file", "--body-data", "--body-file"})
    return [_net("wget", method, _worst(urls), sends)] + _saves("wget", saved)


def _network_tool(base, args) -> list[tuple[str, str]]:
    """Other network tools: method OTHER, except httpie (`http [METHOD] url [items]`: GET, or POST with data items).
    The destination comes from URLs, host:port operands, [user@]host:path (scp, rsync, sftp), socat addresses, or the
    first operand of tools that take a host (ssh, nc, ping, dig, ...); gcloud and gsutil are external. scp and rsync
    to a remote path, and httpie with data items (a=1, a:=1, a@file), send data."""
    if base == "wget":
        return _wget(args)
    _, operands = _opts(args, NET_VALUE_OPTS.get(base, ""), set())
    hosts = [m.group(0) for a in args if (m := URL.search(a))]
    hosts += [a for a in operands if re.fullmatch(r"(\[[^\]]+\]|[\w.-]+):\d+(/\S*)?", a)]
    if base == "socat":
        hosts += [m.group(1) for a in args if (m := re.match(r"(?i)(?:tcp|udp|openssl|ssl)[\w-]*:([^:,]+):\d", a))]
    if base in ("scp", "rsync", "sftp"):
        hosts += [a for a in operands if REMOTE_PATH.match(a)]
    verb = operands[0] if base == "http" and operands and re.fullmatch(r"[A-Z]+", operands[0]) else None  # httpie
    if base in FIRST_OPERAND_HOST:
        hosts += [a for a in operands[bool(verb):] if not a.isdigit() and not a.startswith("+")][:1]
    dest = "external" if base in ("gcloud", "gsutil") else _worst(hosts)
    items = [a for a in operands[1:] if re.match(r"[\w.-]+(:=|=(?!=)|@)", a)] if base == "http" else []
    method = "OTHER" if base != "http" else _http_method(verb) if verb else "POST" if items else "GET"
    sends = bool(items) or (base in ("scp", "rsync") and bool(operands) and bool(REMOTE_PATH.match(operands[-1])))
    return [_net(base, method, dest, sends)]


def _db_host(base, args, whole) -> list[tuple[str, str]]:
    """A network piece when a database tool is given a host: -h, --host, etcdctl --endpoints, a connection URL, a
    conninfo host=, or else PGHOST set in the same command. Socket directories (-h /var/run/postgresql) are local, and
    SQL text (-c, -e, --eval) is not searched."""
    hosts = []
    for k, a in enumerate(args):
        name, eq, value = a.partition("=")
        if a in ("-h", "--host", "--endpoints") and k + 1 < len(args):
            hosts += args[k + 1].split(",")
        elif name in ("--host", "--endpoints") and eq:
            hosts += value.split(",")
        elif re.fullmatch(r"-h[^-=]\S*", a):
            hosts.append(a[2:])
        elif not (k and args[k - 1] in ("-c", "--command", "-e", "--execute", "--eval", "-v", "--set", "--variable",
                                        "-P", "--pset")) and \
                not re.match(r"--(command|execute|eval|set|variable|pset)=|-[cevP]\S", a):  # SQL, variables, settings
            hosts += re.findall(r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|rediss?)(?:\+\w+)?://[^/\s,][^\s,]*", a)
            hosts += re.findall(r"(?:^|\s)host=([^\s'\"]+)", a)
            hosts += [f"${v}" for v in re.findall(r"\$\{?(\w+)\}?", a) if _url_variable(v, whole)]
    if not hosts and base in PG_TOOLS:  # an explicit host or socket directory overrides PGHOST; any PGHOST in the call
        hosts += re.findall(r"\bPGHOST=['\"]?([^\s'\";]+)", whole)  # that names a host counts (errs toward network)
    hosts = [h for h in hosts if h and not h.startswith(("/", "@"))]
    return [_net(f"{base} to a host", "OTHER", _worst(hosts), False)] if hosts else []


def _without_sql(code: str) -> str:
    """Code with SQL string literals and comments blanked, so column names (SELECT database_url ...) are not settings."""
    code = STRINGS.sub(lambda m: "''" if re.search(r"\b(select|insert|update|delete|from|where)\b", m.group(0), re.I)
                       else m.group(0), code)
    return PY_COMMENT.sub("", JS_COMMENT.sub("", code))


def _url_variable(name: str, whole: str) -> bool:
    """A shell variable that holds a connection string: named like one (DATABASE_URL, DB_URI, PG_DSN), or assigned in
    the command from text that names one (DB=$(python3 -c "...['database_url']"), DB=postgres://...)."""
    if re.search(r"url|uri|dsn|conn", name, re.I):
        return True
    m = re.search(rf"(?<![\w$]){re.escape(name)}=", whole)
    if not m:
        return False
    depth, quote, k = 0, None, m.end()
    while k < len(whole):  # the assigned value: up to the first top-level ; & | or newline
        ch = whole[k]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth <= 0 and ch in ";&|\n \t":
            break
        k += 1
    return bool(re.search(r"url|uri|dsn|://", whole[m.end():k], re.I))


def _code_net(code: str, bare: str, origin: str, mods: str = "") -> list[tuple[str, str]]:
    """The network piece of inline code: an HTTP client, another protocol or API client (OTHER), or a database client
    given a host. HTTP methods come from the calls (requests.post, method='PUT', ...); a request that names no method
    (fetch(url), urlopen(url), Request(url)) is GET, one with a body (data=, urlopen(url, body)) is POST, and a method
    in a variable is UNKNOWN. Destinations come from URL literals and host arguments, but not from URLs passed as data
    (params={'next': 'https://...'}): a URL that is the value of a key or keyword not named like url, host or endpoint."""
    http, other = HTTP_CODE.search(bare), OTHER_NET_CODE.search(bare)
    db = (NET_DB_CODE.search(bare) or NET_DB_CODE.search(mods)) and (DB_URL_IN_CODE.search(code) or (
        re.search(r"\bhost(?:name)?\s*[=:]\s*''", bare) and re.search(r"\bhost(?:name)?\s*[=:]\s*['\"](?!/)", code)) or
        DB_URL_NAME.search(_without_sql(code)) or ("PrismaClient" in bare and not re.search(r"(?:datasourceUrl|url)\s*:\s*['\"]file:", code)))
    if not (http or other or db):
        return []
    methods = set()
    if http:  # a method key inside a payload (params={'method': ...}) is data, not the request's method
        methods = {g.upper() for m in CODE_METHOD.finditer(code) for g in m.groups()
                   if g and not (m.group(5) and IN_PAYLOAD.search(code[max(0, m.start() - 80):m.start()]))}
        methods |= {"GET"} if CODE_GET.search(bare) or ("Request(" not in bare and CODE_URLOPEN_GET.search(bare)) else set()
        methods |= {"POST"} if URLLIB_BODY.search(bare) else set()
        methods |= {"UNKNOWN"} if CODE_DYNAMIC_METHOD.search(code) or CODE_DYNAMIC_FETCH.search(bare) else set()
    methods = {m if m in HTTP_METHODS | {"UNKNOWN"} else "OTHER" for m in methods} | ({"OTHER"} if other or db else set())
    hosts = []
    for m in CODE_HOST.finditer(code):
        key = URL_KEY.search(code[max(0, m.start() - 60):m.start()]) if m.group(1) else None
        if not (key and not re.search(r"url|uri|endpoint|host|addr|server|base|target", key.group(1), re.I)):
            hosts += [g for g in m.groups() if g]
    method = min(methods, key=METHODS.index) if methods else "UNKNOWN"
    sends = bool(CODE_SENDS.search(code) or URLLIB_BODY.search(bare))
    return [_net(f"{origin} code", method, _worst(hosts), sends)]


def _script(path: str) -> tuple[str, str]:
    name = path.rsplit("/", 1)[-1]
    if SERVICE_NAME.search(name):
        return ("process_service", f"service script {name[:40]}")
    if HELPER_READ.search(name.removesuffix(".sh")) and not path.startswith(("/tmp/", "/dev/shm/")) and not name.startswith("."):
        return ("inspect", f"read helper {name[:40]}")
    return ("run_program", f"script {name[:40]}")


def _find(args, whole, defined):
    if "-delete" in args:
        return [("file_delete", "find -delete")]
    if {"-fprint", "-fprintf", "-fls"} & set(args):
        return [("file_write", "find -fprint")]
    found = []
    for k, a in enumerate(args):  # every -exec clause
        if a in ("-exec", "-execdir", "-ok", "-okdir"):
            inner = []
            for b in args[k + 1:]:
                if b in (";", "\\;", "+"):
                    break
                inner.append(b)
            found += [(c, f"find -exec > {r}") for c, r in _label_words(inner, [], whole, defined)] if inner else []
    return [f for f in found if f[0] not in READ_ONLY] or [("file_read", "find")]


def _git(args) -> tuple[str, str]:
    rest = list(args)
    while rest and (rest[0].startswith("-") or ASSIGN.match(rest[0])):
        opt = rest.pop(0)
        if opt in ("-C", "-c", "--git-dir", "--work-tree") and rest:
            rest.pop(0)
    sub, more = (rest[0], rest[1:]) if rest else ("", [])
    listing = {"branch": {"-a", "-r", "-v", "-vv", "--list", "--show-current", "-l"}, "remote": {"-v", "show", "get-url"},
               "tag": {"-l", "--list"}, "stash": {"list", "show"}, "config": {"--get", "--list", "-l", "--get-all"}}
    bare_lists = {"branch", "remote", "tag"}  # bare `git stash` saves and resets changes
    if sub in GIT_READ or (sub in listing and ((not more and sub in bare_lists) or set(more) & listing[sub])):
        return ("file_read", f"git {sub}")
    if sub in ("push", "pull", "fetch", "clone", "ls-remote", "submodule"):  # a remote name hides its host
        hosts = [a for a in more if URL.search(a) or (REMOTE_PATH.match(a) and "@" in a.split(":")[0])]  # git@host:x
        return _net(f"git {sub}", "OTHER", _worst(hosts), sub == "push")
    if sub in ("rm", "clean"):
        return ("file_delete", f"git {sub}")
    return ("file_write", f"git {sub or 'other'}")


def _admin(base, args) -> tuple[str, str]:
    reads = {"sysctl": "-w" not in args, "mount": not args, "ldconfig": "-p" in args, "crontab": not args or "-l" in args,
             "openssl": not ({"-out", "s_client"} & set(args)) and (args[:1] != ["genrsa"]),
             "update-alternatives": bool({"--display", "--query", "--list", "--get-selections"} & set(args)),
             "dpkg": not ({"-i", "--install", "-r", "--remove", "-P", "--purge", "--configure"} & set(args))}
    if reads[base]:
        return ("inspect", f"{base} (read)")
    if base == "openssl" and "s_client" in args:
        host = args[args.index("-connect") + 1] if "-connect" in args[:-1] else ""
        return _net("openssl s_client", "OTHER", _dest(host), False)
    return ({"openssl": "file_write", "dpkg": "install"}.get(base, "permissions_config"), base)


def _build_tool(base, args) -> tuple[str, str]:
    sub = next((a for a in args if not a.startswith("-")), "")
    second = next((a for a in args[args.index(sub) + 1:] if not a.startswith("-")), "") if sub else ""
    installs = {"npm": "install i ci add remove uninstall rm update upgrade link", "pnpm": "install i add remove update",
                "yarn": "install add remove upgrade", "bun": "install i add remove update", "composer": "install require "
                "update remove", "bundle": "install add update remove", "go": "get install", "cargo": "install add "
                "remove update fetch uninstall", "uv": "add remove sync lock venv tool", "poetry": "install add remove "
                "update lock", "pipenv": "install uninstall update lock sync", "dotnet": "add restore new tool",
                "deno": "install add cache", "npx": "", "prisma": "", "flask": "", "mix": "deps.get deps.update"}
    if sub in installs[base].split() or (base == "yarn" and not sub) or (base == "bundle" and not sub) or \
            (base == "go" and sub == "mod" and second in ("tidy", "download", "vendor", "init")) or \
            (base == "uv" and sub == "pip" and second in ("install", "uninstall", "sync", "compile")):
        return ("install", f"{base} {sub}".strip())
    if base in ("cargo", "go") and sub == "fmt":
        return ("file_read", f"{base} fmt --check") if "--check" in args else ("file_write", f"{base} fmt")
    if base == "npx" and "tsc" in args:  # npx tsc ...: the tsc rule
        return _label_command(args[args.index("tsc"):], [], "", set())[0]
    if base in ("npx", "prisma") and "validate" in args:  # schema check
        return ("file_read", f"{base} prisma validate")
    if base == "flask" and sub == "run":
        return ("process_service", "flask run")
    if sub in ("ls", "list", "view", "info", "outdated", "why", "tree", "metadata", "version", "env", "doc", "config",
               "--info", "--list-sdks") or (base == "uv" and sub == "pip") or (base == "bundle" and sub == "check" and
                                                                             "--dry-run" in args):  # else may write Gemfile.lock
        return ("inspect", f"{base} {sub}")
    return ("run_program", f"{base} {sub}".strip())


def _scan_code(code: str, origin: str) -> list[tuple[str, str]]:
    bare = STRINGS.sub("''", code)  # API names inside string literals (edited source, messages) are not calls
    spans = [m.span() for m in STRINGS.finditer(code)]
    in_string = lambda k: any(a <= k < b for a, b in spans)
    spans += [m.span() for m in JS_COMMENT.finditer(code) if not in_string(m.start())]  # // and /* */ outside strings
    modules = [m.group(1) or m.group(2) for m in MODULE_NAME.finditer(code)  # require('pg'), import ... from 'pg'
               if not any(a <= m.start() < b for a, b in spans)]  # written as code: not in a string or a comment
    mods = " ".join(modules)  # used only to recognize database clients, not by the other rules
    found = [(cat, f"{origin} code") for cat, pat in CODE_RULES if re.search(pat, code if cat == "file_write" else bare)]
    if not any(c == "file_write" for c, _ in found) and _serializer_writes(code):
        found.append(("file_write", f"{origin} code"))
    # A /tmp-only write: some write has a literal target, and every absolute path the code names is under /tmp
    # (a variable holding /app/... would show up as a literal elsewhere).
    paths = [x for x in re.findall(r"['\"](/[^'\"\s]*)['\"]", code) if not x.startswith("/dev/")]
    if ("file_write", f"{origin} code") in found and WRITE_TARGET.search(code) and paths and \
            all(x.startswith(("/tmp/", "/dev/shm/")) for x in paths):
        found = [("file_write_tmp", f"{origin} code writes /tmp") if f == ("file_write", f"{origin} code") else f for f in found]
    edits_files = any(c == "file_write" for c, _ in found) and ".replace(" in code  # SQL text is likely edited source
    db_lib = DB_IN_CODE.search(bare) or DB_IN_CODE.search(mods)
    if db_lib and _sql_write_reaches_db(code, origin) and not edits_files:
        found.append(("db_write", f"{origin} code SQL write"))
    elif ORM_WRITE.search(bare) and db_lib and not edits_files:
        found.append(("db_write", f"{origin} code ORM write"))
    for m in re.finditer(r"(?:check_output|check_call|run|call|Popen|system|execSync|spawnSync|popen)\(\s*\[?\s*f?(['\"])"
                         r"(.+?)\1", code):  # literal subprocess commands
        found.extend((c, f"{origin} subprocess > {r}") for c, r in label_shell(m.group(2)) if c not in READ_ONLY)
    net = _code_net(code, bare, origin, mods)
    if not found and origin.startswith("python"):
        mods = {m.strip().split(".")[0] for a, b in PY_IMPORT.findall(code) for m in (a or b).split(",")}
        if mods - sys.stdlib_module_names - PY_LIBS_READ - (PY_NET_LIBS if net else set()):
            found.append(("run_program", f"{origin} imports project code"))
    if re.match(r"(node|deno|bun|ts-node|tsx)\b", origin):  # a JavaScript require of a .json file reads data
        code = re.sub(r"require\(\s*['\"][^'\"]*\.json['\"]\s*\)", "JSON", code)
    if not found and PROJECT_CODE.search(code):  # module paths are string literals
        found.append(("run_program", f"{origin} loads project code"))
    if not found and db_lib:
        found.append(("db_read", f"{origin} code queries a database"))
    # Added after the checks above, so network code that also loads project code keeps run_program: a narrower network
    # rule recomputed from the categories must not make it read-only.
    found += net
    return found or [("inline_code_read", f"{origin} code, no side effects found")]


def _sql_write_reaches_db(code: str, origin: str = "") -> bool:
    """True if a string with SQL write text, outside comments, is sent to a database: it is an argument of a database
    call (client.query('INSERT ...'), $pdo->exec(...), dbExecute(con, 'UPDATE ...'), a tagged sql`...` template), or
    it is assigned to a name that a database call takes (q = 'INSERT ...'; cur.execute(q); $q = ...; $pdo->exec($q)).
    SQL that is only printed, commented out or stored next to an unrelated call does not count. Comments follow the
    language: // and /* */ in JavaScript and PHP, # in Python, Ruby, R, Perl and PHP."""
    strings = [m for m in STRINGS.finditer(code)]
    if origin.startswith("ruby"):  # Ruby's %q{...} literals, which may hold quotes of their own
        pct = list(RUBY_PERCENT.finditer(code))
        strings = pct + [m for m in strings if not any(q.start() <= m.start() < q.end() for q in pct)]
    in_string = lambda k: any(m.start() <= k < m.end() for m in strings)
    js = re.match(r"(node|deno|bun|ts-node|tsx|php)\b", origin) is not None
    hash_comments = not re.match(r"(node|deno|bun|ts-node|tsx)\b", origin)
    comments = [m.span() for pat, on in ((JS_COMMENT, js), (PY_COMMENT, hash_comments)) if on
                for m in pat.finditer(code) if not in_string(m.start())]
    calls = [(m.end() - 1, _close(code, m.end() - 1)) for m in DB_CALL.finditer(code)  # (open, close paren) of each
             if not in_string(m.start()) and not any(a <= m.start() < b for a, b in comments)]  # live database call
    bare = "".join(" " * len(ch) if in_string(k) else ch for k, ch in enumerate(code))  # names outside strings
    for m in strings:
        if not SQL_WRITE.search(m.group(0)) or any(a <= m.start() < b for a, b in comments):
            continue
        if any(o < m.start() < c for o, c in calls) or re.search(r"\bsql\s*$", code[max(0, m.start() - 10):m.start()]):
            return True
        bound = re.search(r"(\$?\b\w+)\s*=\s*[frbFRB]{0,2}$", code[max(0, m.start() - 80):m.start()])
        if bound and any(re.search(rf"(?<![\w$]){re.escape(bound.group(1))}\b", bare[o + 1:c]) for o, c in calls):
            return True
    return False


def _call_args(code: str, open_paren: int) -> list[str]:
    """Top-level arguments of the call whose "(" is at code[open_paren]."""
    end = _close(code, open_paren)
    args, depth, quote, start, escaped = [], 0, None, open_paren + 1, False
    for k in range(open_paren + 1, end):
        ch = code[k]
        if escaped:
            escaped = False
        elif quote and ch == "\\":
            escaped = True
        elif quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(code[start:k].strip())
            start = k + 1
    last = code[start:end].strip()
    return args + [last] if last else args  # a trailing comma adds no argument


def _serializer_writes(code: str) -> bool:
    """True if a pandas to_csv/to_json/to_parquet or an R write.csv/write.table call names a file. With no path (or
    None, or stdout) pandas returns a string and R prints; neither writes a file."""
    stdout = re.compile(r"^(None|sys\.stdout|stdout\(\)|\"\"|'')$")
    for m in SERIALIZER.finditer(code):
        args = _call_args(code, m.end() - 1)
        named = {a.split("=", 1)[0].strip(): a.split("=", 1)[1].strip() for a in args if re.match(r"^[\w.]+\s*=[^=]", a)}
        positional = [a for a in args if not re.match(r"^[\w.]+\s*=[^=]", a)]
        if m.group(1):  # pandas: the first positional argument or path_or_buf / path
            dest = positional[0] if positional else named.get("path_or_buf", named.get("path"))
        else:  # R: write.csv(x, file = "") prints. Named arguments bind first; positional ones fill x, then file.
            piped = re.search(r"(\|>|%>%)\s*(\w+::)?$", code[:m.start()])  # df |> utils::write.csv(path): x is piped
            free = [f for f in ("x", "file") if f not in named and not (f == "x" and piped)]
            dest = named.get("file") or dict(zip(free, positional)).get("file")
        if dest and not stdout.match(dest):
            return True
    return False


def _interpreter(base, args, heredocs) -> list[tuple[str, str]]:
    name = "python" if base.startswith("python") else base
    if base == "perl" and any(re.match(r"^-[a-zA-Z]*i", a) for a in args):
        return [("file_write", "perl -i")]
    code_flags, check_flags, takes_value = INTERPRETER_OPTS.get(name, ({"-e", "--eval"}, set(), set()))
    preload = []  # local modules loaded by options (ruby -r./app, node --require ./x.js) run project code
    loadpath = []

    def scan(code, origin):
        found = _scan_code(code, origin)
        if loadpath and re.search(r"\brequire\s*\(?\s*['\"]|\buse\s+[A-Z]", code):  # ruby -Iapp/lib -e "require 'x'"
            preload.append(f"from {loadpath[0]}")
        if preload:
            found = [f for f in found if f[0] != "inline_code_read"] + [("run_program", f"{name} preloads {preload[0][:40]}")]
        return found

    k = 0
    while k < len(args):
        a = args[k]
        attached = len(a) > 2 and a[:2] in takes_value  # -rjson, -Mstrict, -Wignore
        flag, _, value = a.partition("=") if a.startswith("--") else (a[:2], "", a[2:]) if attached else (a, "", "")
        value = value or (args[k + 1] if a in takes_value and k + 1 < len(args) else "")
        if flag in ("-r", "--require", "--import", "--loader") and "/" in value:
            preload.append(value)
        if flag == "-I" and name in ("ruby", "perl"):  # a load path: requires may load code from it
            loadpath.append(value)
        combined = re.fullmatch(r"-[a-zA-Z]{2,}", a) and not attached and a[-1] in {f[1] for f in code_flags if len(f) == 2}
        if a in code_flags or combined:  # -c, -e, and bundles such as -ne or -uc
            return scan((args[k + 1] if k + 1 < len(args) else "") + "\n" + "\n".join(heredocs), f"{name} {a}")
        if a in check_flags:
            return [("file_read", f"{name} {a} (syntax check)")]
        if a == "-m" and name == "python":
            mod = args[k + 1] if k + 1 < len(args) else ""
            if mod in ("json.tool", "pydoc", "site", "platform", "sysconfig", "tokenize", "dis", "tabnanny",
                       "py_compile", "compileall"):  # syntax checks are read-only (Aswin's hand-check)
                return [("file_read", f"python -m {mod}")]
            if mod in ("pip", "venv", "virtualenv", "ensurepip"):
                verb = next((x for x in args[k + 2:] if not x.startswith("-")), "")
                return [("inspect", f"python -m pip {verb}")] if mod == "pip" and verb in ("list", "show", "freeze") else \
                    [("install", f"python -m {mod} {verb}".strip())]
            if mod in ("http.server", "uvicorn", "gunicorn", "flask", "celery", "smtpd", "aiosmtpd"):
                return [("process_service", f"python -m {mod}")]
            return [("run_program", f"python -m {mod}")]
        if a == "-":
            break
        if not a.startswith("-"):
            return [("run_program", f"{name} script")]
        k += 2 if a in takes_value else 1
    if heredocs:
        return scan("\n".join(heredocs), f"{name} stdin")
    return [("run_program", f"{name} stdin or REPL")]


def _db(base, args, heredocs, whole) -> tuple[str, str]:
    if base == "redis-cli":
        if {"--scan", "--bigkeys", "--memkeys", "--stat", "--latency", "--hotkeys"} & set(args):
            return ("db_read", "redis-cli scan/stat mode")
        k = 0
        while k < len(args) and args[k].startswith("-"):
            k += 1 if args[k] in ("--tls", "-x", "--raw", "--no-raw", "-c", "--no-auth-warning", "-e") else 2
        verb = args[k].lower() if k < len(args) else ""
        if not verb:  # commands come from a heredoc or a pipe: look for upper-case Redis write commands at line starts
            source = "\n".join(heredocs) or whole
            writes = "|".join(sorted((w.upper() for w in REDIS_WRITE if "." not in w), key=len, reverse=True))
            return ("db_write", "redis-cli stdin write") if re.search(rf"(?m)(^|['\"]|\\n)\s*({writes})\b", source) else \
                ("db_read", "redis-cli stdin")
        return ("db_write", f"redis-cli {verb}") if verb in REDIS_WRITE else ("db_read", f"redis-cli {verb}")
    if base == "etcdctl":
        verb = next((a for a in args if not a.startswith("-")), "")
        return ("db_write", f"etcdctl {verb}") if verb in ("put", "del", "txn", "lease", "lock", "user", "role",
                                                            "auth", "member", "defrag", "snapshot") else \
            ("db_read", f"etcdctl {verb}")
    text = " ".join(args) + "\n" + "\n".join(heredocs)
    if base in ("mongosh", "mongo"):
        return ("db_write", f"{base} write") if ORM_WRITE.search(text) or re.search(
            r"\.(insert|drop|remove|update|createIndex|createCollection)\w*\(", text) else ("db_read", base)
    # SQL from -c/-e, a heredoc, or another piece of the same call (cat <<SQL | psql, printf '...' | psql).
    inline = heredocs or any(a in ("-c", "-e", "--command", "--execute") or re.fullmatch(r"-\w*c", a) for a in args)
    if SQL_WRITE.search(text) or (not inline and SQL_WRITE.search(whole)):
        return ("db_write", f"{base} SQL write")
    if {"-f", "--file"} & set(args):
        return ("run_program", f"{base} -f (SQL file)")
    if inline or {"-l", "--list"} & set(args):
        return ("db_read", f"{base} SQL read")
    if re.search(r"\.(sql|psql|dump)\b|<\s*[^<\s]", whole) or not re.search(r"(?i)\b(select|show|explain|with)\b|\\d", whole):
        return ("run_program", f"{base} SQL from a file or stdin")  # the statements are not visible
    return ("db_read", f"{base} SQL read from a pipe")
