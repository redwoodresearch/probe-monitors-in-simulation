def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def msg_sha(m) -> str:
    return sha(m.content) if isinstance(m.content, str) else "list:" + sha(m.text)


def hash_eval(job: tuple[str, str]) -> dict:
    path, policy = job
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    # resolve_attachments: events store long texts as attachment://<hash> references, which would be hashed instead
    (s,) = read_eval_log(path, resolve_attachments=True).samples
    calls = [e for e in s.events if isinstance(e, ModelEvent) and e.model == policy]
    sent = calls[0].input if calls else []
    approach = s.metadata.get("attack_approach")
    return {"approach_sha256": None if approach is None else sha(approach),
            "policy_calls": len(calls),
            "sent_roles": [m.role for m in sent[:2]],
            "sent_system_sha256": msg_sha(sent[0]) if sent and sent[0].role == "system" else None,
            "sent_user_sha256": msg_sha(sent[1]) if len(sent) > 1 and sent[1].role == "user" else None,
            # the first call in full, lengths and hashes only: every message (role and text), every user message's
            # length, the first non-empty user message, and the tool definitions sent with it
            "sent_n_messages": len(sent),
            "sent_system_len": len(sent[0].text) if sent and sent[0].role == "system" else None,
            "sent_user_lens": [len(m.text) for m in sent if m.role == "user"],
            "sent_first_nonempty_user": next(({"index": i, "sha256": msg_sha(m)} for i, m in enumerate(sent)
                                              if m.role == "user" and m.text.strip()), None),
            "sent_all_sha256": sha(json.dumps([[m.role, m.text] for m in sent])),
            "sent_tools_sha256": sha(json.dumps([t.model_dump() for t in calls[0].tools], sort_keys=True, default=str))
            if calls else None}

