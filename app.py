"""Free hosted demo. No local model server, secrets or shared visitor files."""
import json
from datetime import datetime, timezone
import streamlit as st
from src.settings import DATA
from src.engine import PROFILES, rank, deadline_status, available_for_research, checks, brief_html
from src.ai import validate_extraction

def read(name):
    return json.loads((DATA/name).read_text(encoding="utf-8"))

st.set_page_config(page_title="Grant Intelligence | Fellowship prototype",page_icon="🌱",layout="wide")
st.caption("CHIEAC GRANT INTELLIGENCE / INDEPENDENT FELLOWSHIP PROTOTYPE")
st.title("Find funding that fits.")
st.write("Explore funding sources, check restrictions and download a research brief.")
st.caption("Prepared for feedback. Organizational approval and funding outcomes are not implied.")
page=st.sidebar.radio("Workspace",["Discover","AI example","Reviewer feedback","About & evaluation"])
records=read("local_grants.json")
program=st.sidebar.selectbox("Program",list(PROFILES))
st.sidebar.caption("Program descriptions are drafts awaiting ChiEAC confirmation.")
st.sidebar.info("Free demo: no account or paid AI key required. Feedback stays in your browser session; download it before leaving.")

if page=="Discover":
    st.info("Start here: choose a program, inspect a result, open its original source, then download a brief.")
    method=st.selectbox("Matching method",["Keyword search (live)","Neural matching (precomputed)"])
    neural=method.startswith("Neural")
    query=st.text_area("Program description",PROFILES[program],key="query_"+program+str(neural),disabled=neural,max_chars=3000)
    if neural:
        st.caption("Actual nomic-embed-text scores computed on October 8, 2026 for these fixed program descriptions and 14 records. No live model call occurs.")
        snapshot=read("neural_snapshot.json")
        scored=[dict(r,similarity=snapshot["scores"][program][r["id"]]) for r in records]
        ranked=sorted(scored,key=lambda r:-r["similarity"])
    else:
        ranked=rank(query,records,"Keyword baseline")
    hide=st.checkbox("Show currently available research routes only",value=True)
    if hide:
        ranked=[r for r in ranked if available_for_research(r)]
    st.caption("Source snapshot: October 7, 2026. Summaries are assistant-written paraphrases. Verify every source; similarity is not eligibility or award probability.")
    st.metric("Results",len(ranked))
    if not ranked:
        st.warning("No results. Enter a description or turn off the availability filter.")
    else:
        for i,r in enumerate(ranked[:5],1):
            with st.container(border=True):
                st.markdown(f"**{i}. {r['title']}**")
                st.caption(r["funder"]+" | "+deadline_status(r)+f" | Similarity {r['similarity']:.3f}")
        choices={r["title"]+" ["+r["id"]+"]":r for r in ranked}
        r=choices[st.selectbox("Inspect opportunity",list(choices))]
        st.subheader(r["title"])
        st.write(r["text"])
        st.warning(r["review_notes"])
        st.link_button("Read original funder source",r["url"])
        for i,url in enumerate(r.get("additional_sources",[])):
            st.link_button("Supporting source "+str(i+1),url)
        st.write("Deadline / review cutoff:",r.get("deadline") or "Not confirmed")
        rows=checks(r)
        st.dataframe(rows,hide_index=True,use_container_width=True)
        st.download_button("Download research brief",brief_html(r,query,rows,method),file_name=r["id"]+"-brief.html",mime="text/html")
        st.caption("Open the downloaded HTML in a browser; Print > Save as PDF to share it.")

elif page=="AI example":
    st.subheader("See what the local document model produced")
    st.warning("Recorded demonstration using fictional grant text. This page does not run live AI extraction. The desktop version runs Llama through Ollama.")
    example=read("extraction_example.json")
    st.text_area("Fictional source text",example["source_text"],height=180,disabled=True)
    st.caption("Recorded model: "+example["model"])
    result=validate_extraction(example["result"],example["source_text"])
    st.dataframe(result["findings"],hide_index=True,use_container_width=True)
    st.write("The source check confirms that the words occur in the document. It does not confirm the category or completeness: this example includes a document requirement classified as eligibility.")
    st.download_button("Download recorded example",json.dumps(example,indent=2),file_name="recorded-ai-example.json",mime="application/json")

elif page=="Reviewer feedback":
    st.subheader("Help evaluate topical fit")
    st.write(PROFILES[program])
    st.caption("Scores are hidden on this page. Review original sources before assigning a grade. Do not include sensitive information.")
    options={r["title"]+" ["+r["id"]+"]":r for r in records}
    r=options[st.selectbox("Opportunity to review",list(options))]
    st.link_button("Read source",r["url"])
    for i,url in enumerate(r.get("additional_sources",[])):
        st.link_button("Supporting source "+str(i+1),url)
    st.write(r["text"])
    st.session_state.setdefault("feedback",{})
    with st.form("feedback_"+program+r["id"]):
        alias=st.text_input("Reviewer name or alias",max_chars=100)
        grade=st.selectbox("Topical relevance",["Choose a grade","0 - Unrelated","1 - Conditional / adjacent","2 - Direct fit"])
        note=st.text_area("Reason and source corrections",max_chars=3000)
        confirmed=st.checkbox("I reviewed the original source")
        submit=st.form_submit_button("Save to this session")
    if submit:
        if not alias.strip() or grade=="Choose a grade" or not note.strip() or not confirmed:
            st.error("Enter a reviewer alias, grade and reason, then confirm the source check.")
        else:
            st.session_state["feedback"][program+"::"+r["id"]]={"program":program,"query":PROFILES[program],"id":r["id"],"source":r["url"],"grade":int(grade[0]),"reviewer":alias.strip(),"notes":note.strip(),"reviewed_at":datetime.now(timezone.utc).isoformat()}
            st.success("Saved for this session. Download below to keep or share your feedback.")
    st.caption(str(len(st.session_state["feedback"]))+" of 42 program/opportunity pairs reviewed in this session.")
    st.download_button("Download my feedback",json.dumps(st.session_state["feedback"],indent=2),file_name="grant-review-feedback.json",mime="application/json")
    st.caption("Nothing is written to a shared review file. Reloading or disconnecting can lose your session. Send the downloaded file to the fellow if you want it included in future evaluation.")

else:
    st.subheader("What this prototype is testing")
    st.write("Can source-linked funding research and AI-assisted document review reduce the time needed to prepare useful funding briefs?")
    st.write("This hosted demo includes 14 research records, live keyword matching, precomputed neural comparisons, brief exports and session-only feedback. Live Grants.gov importing and local Llama document extraction remain in the desktop version.")
    payload=read("evaluation.json")
    st.warning(payload["scope"])
    st.dataframe(payload["summary"],hide_index=True,use_container_width=True)
    st.caption("These metrics describe the October 7 development benchmark, not independent human validation or the probability of receiving funding.")
    st.download_button("Download benchmark details",json.dumps(payload,indent=2),file_name="provisional-benchmark.json",mime="application/json")
    st.write("Suggested pilot: confirm program descriptions, review relevance, and measure time saved and errors found.")
