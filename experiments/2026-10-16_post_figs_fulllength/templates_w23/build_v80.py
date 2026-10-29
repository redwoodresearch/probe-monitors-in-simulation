import re,sys,base64,markdown,os
S=sys.argv[1]; name=sys.argv[2]
md=open(f"{S}/{name}.md").read()
md=md.replace("[TABLE 4]", open(f"{S}/appendixJ_table.md").read().replace("GPT-5.6-Sol probe","Off-policy probe"))
rw=f"{S}/related_work.md"
md=md.replace("[RELATED WORK]", open(rw).read() if os.path.exists(rw) else "[COMMENT: related work list pending.]")
md=re.sub(r"<!--.*?-->","",md,flags=re.S)
md=re.sub(r"^\[(TODO[^:]*): (.*)\]$", lambda m: f'<p class="c"><b>{m.group(1)}:</b> {m.group(2)}</p>', md, flags=re.M)
html=markdown.markdown(md, extensions=["footnotes","tables","fenced_code"])
for n,f in {1:"fig1_pairs_all.png",2:"fig0_method.png",3:"fig2_excess_C.png",4:"fig_rms.png",5:"fig_pertask_onpolicy.png",6:"fig_sweep_final.png",7:"fig1_pairs_all_full.png",8:"fig_pertask.png",9:"fig_pertask_summary.png",10:"fig1_pairs.png",11:"fig2_excess_combined.png"}.items():
    b=base64.b64encode(open(f"{S}/figs_simple/{f}","rb").read()).decode()
    html=html.replace(f"<p>[FIGURE {n}]</p>",f'<p><img alt="Figure {n}" style="max-width:100%" src="data:image/png;base64,{b}"></p>')
page=f"""<!doctype html><html><head><meta charset="utf-8"><title>Probe-shift draft {name}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet"><style>body{{font-family:"EB Garamond",Georgia,serif;font-size:19px;max-width:780px;margin:24px auto;padding:0 16px;line-height:1.6;color:#222;background:#fff}} code{{background:#f3f3f1;padding:1px 4px}} table{{border-collapse:collapse;margin:12px 0;font-size:13px}} td,th{{border:1px solid #ccc;padding:4px 6px;text-align:left}} .c{{background:#fff3c4;border-left:3px solid #e0a800;padding:6px 10px;font-family:sans-serif;font-size:14px}}</style></head><body>{html}</body></html>"""
open(f"{S}/{name}.html","w").write(page); print("bytes",len(page),"unreplaced:",page.count("[FIGURE"))
