"""
GPT-5-mini verifier over the NON-REGENERATED C1 union (combined_sft_c1 10,236 + grpo 4,236 = 14,472
unique traces; the set the 6K is a 100% subset of). Sharded for parallel tmux runs.
Run one shard:  python verify_c1_union.py --shard <0-7> --nshards 8
Shards write to a SHARED dir with skip-existing, so they never collide and are resumable.
"""
import os, io, re, json, time, base64, argparse, logging
from pathlib import Path
from PIL import Image
from openai import OpenAI

ROOT = Path("/bulk/aacudad/reasoning_traces")
ENV_PATH = ROOT / "repository_tu_delft_vlms/.env"
UNION = ROOT / "outputs/c1_union_dedup.json"            # [{image_id, image, reasoning, src}]
OUT = ROOT / "outputs/verify_c1_union_results"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gpt-5-mini"; BATCH = 15

VALID_ISSUES = ["missing_think_tags","missing_answer_tag","missing_location_tag","missing_type_tag",
                "extra_location_tag","extra_type_tag","has_batch_reference","illogical_reasoning",
                "incoherent_content","wrong_product_reference"]

SYS = f"""You are an expert verifier auditing industrial-defect reasoning traces for a TRAINING-DATA QUALITY FILTER. You grade each trace against the IMAGES, never against how fluent or confident it sounds.

For EACH trace you receive the ORIGINAL product image, then (for anomalies) an OVERLAY image with the ground-truth defect region in RED, then the reasoning text. Normal samples have no overlay.

Grade each CRITERION as MET/UNMET (judge on the image, not the prose), then assign a status.
FORMAT: F1 has <think>...</think>; F2 exactly one <answer>Yes/No</answer>; F3 Yes=>has <location>&<type>, No=>has neither.
CONTENT (vs the images): C1 correct product; C2 (anomaly) described defect matches what is visible in the RED region (not hallucinated/different); C3 (anomaly) <location> matches where the red region actually is; C4 visible cues described BEFORE naming the defect/verdict; C5 steps coherent and actually support the conclusion; C6 no batch/sequence words ("another/next/previous/final/tenth/across samples/in this batch") and no mention of labels/masks/overlay/ground-truth; C7 causes are tentative not asserted as fact; C8 (normal) inspects key regions and justifies "No" against absence of cues, not a generic "looks fine".
NOT criteria (never flag): trace length/verbosity, stylistic wording, exact location phrasing if right area, exact type naming if it matches what's visible.

A fluent, confident trace that names a defect not visible in the red region, or in the wrong place, MUST fail C2/C3. Leakage (C6), wrong product (C1), or illogical reasoning (C5) is AT MOST "wrong" even if otherwise excellent.

STATUS: completely_correct = all FORMAT + all applicable CONTENT met. correct = all FORMAT met, C1/C2/C3/C5/C6 met, at most a minor C4/C7/C8 slip. wrong = any FORMAT violation OR any of C1/C2/C3/C5/C6 UNMET. completely_wrong = multiple/severe failures.

ISSUE CODES (pick ONLY from this list; EMPTY for completely_correct/correct):
{json.dumps(VALID_ISSUES)}

Return strict JSON: {{"verifications":[{{"image_id":"exact_id","issues":[...],"issue_elaboration":["Image shows X but reasoning says Y", ...],"status":"completely_correct|correct|wrong|completely_wrong","confidence_score":0.0-1.0}}]}}. Write issues+elaboration FIRST then status. One entry per trace. confidence is an audit signal only."""

def load_key():
    for ln in ENV_PATH.read_text().splitlines():
        if ln.strip().startswith("OPENAI_API_KEY"): return ln.split("=",1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key")

def b64(im):
    b=io.BytesIO(); im.convert("RGB").save(b,format="JPEG",quality=85); return base64.b64encode(b.getvalue()).decode()

def overlay(img_path, mask_path):
    base=Image.open(img_path).convert("RGBA"); m=Image.open(mask_path).convert("L")
    if m.size!=base.size: m=m.resize(base.size, Image.NEAREST)
    red=Image.new("RGBA",base.size,(0,0,0,0)); px=red.load(); mp=m.load()
    for x in range(base.width):
        for y in range(base.height):
            if mp[x,y]>127: px[x,y]=(255,0,0,128)
    return Image.alpha_composite(base,red).convert("RGB")

def verify_batch(client, batch):
    content=[{"type":"input_text","text":f"Verify these {len(batch)} traces."}]
    for i,it in enumerate(batch,1):
        content.append({"type":"input_text","text":f"\n--- TRACE {i} ---\nID: {it['image_id']}"})
        try:
            if os.path.exists(it["image"]):
                content.append({"type":"input_image","image_url":f"data:image/jpeg;base64,{b64(Image.open(it['image']))}"})
                if os.path.exists(it["mask"]):
                    content.append({"type":"input_image","image_url":f"data:image/jpeg;base64,{b64(overlay(it['image'],it['mask']))}"})
        except Exception:
            content.append({"type":"input_text","text":"[image load failed]"})
        content.append({"type":"input_text","text":"\nReasoning:\n"+it["reasoning"]})
    r=client.responses.create(model=MODEL,
        input=[{"role":"system","content":SYS},{"role":"user","content":content}],
        reasoning={"effort":"low"}, text={"format":{"type":"json_object"}})
    txt=None
    for o in r.output:
        if getattr(o,"type",None)=="message":
            for c in o.content:
                if getattr(c,"type",None)=="output_text": txt=c.text
    js=json.loads(txt)
    if isinstance(js,list): js={"verifications":js}
    return {v["image_id"]:v for v in js.get("verifications",[])}, r.usage

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int,required=True); ap.add_argument("--nshards",type=int,default=8); a=ap.parse_args()
    PROG=OUT/f"progress_shard{a.shard}.txt"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s "+f"[shard{a.shard}] "+"%(message)s",
                        handlers=[logging.FileHandler(OUT/f"run_shard{a.shard}.log"), logging.StreamHandler()], force=True)
    log=logging.getLogger(); client=OpenAI(api_key=load_key())
    allitems=json.load(open(UNION))
    items=allitems[a.shard::a.nshards]   # round-robin slice (mixed products)
    for it in items: it["mask"]=it["image"].rsplit(".",1)[0]+".png"
    done=set()
    for p in OUT.glob("verify_*.json"):
        try:
            if json.load(open(p)).get("status")!="error": done.add(p.stem.replace("verify_",""))
        except Exception: pass
    todo=[it for it in items if it["image_id"] not in done]
    log.info(f"shard {a.shard}/{a.nshards}: items={len(items)} done={len(items)-len(todo)} todo={len(todo)}")
    tin=tout=0; counts={}
    batches=[todo[i:i+BATCH] for i in range(0,len(todo),BATCH)]
    for bi,batch in enumerate(batches):
        try:
            res=u=None
            for att in range(3):
                try: res,u=verify_batch(client,batch); break
                except Exception as ie:
                    if att==2: raise
                    log.info(f"batch {bi} retry {att+1}: {str(ie)[:100]}"); time.sleep(8*(att+1))
            tin+=u.input_tokens; tout+=u.output_tokens
            for it in batch:
                v=res.get(it["image_id"], {"image_id":it["image_id"],"status":"error","issues":[],"issue_elaboration":[],"confidence_score":0.0})
                v["src"]=it.get("src")
                json.dump(v, open(OUT/f"verify_{it['image_id']}.json","w"), indent=1)
                counts[v.get("status","error")]=counts.get(v.get("status","error"),0)+1
        except Exception as e:
            log.error(f"batch {bi} FAILED: {str(e)[:150]}")
            for it in batch:
                json.dump({"image_id":it["image_id"],"status":"error","issues":[],"issue_elaboration":[],"confidence_score":0.0,"src":it.get("src")}, open(OUT/f"verify_{it['image_id']}.json","w"))
                counts["error"]=counts.get("error",0)+1
        cost=tin/1e6*0.25+tout/1e6*2.0
        msg=f"batch {bi+1}/{len(batches)} ~${cost:.2f} {counts}"; PROG.write_text(msg); log.info(msg); time.sleep(0.3)
    PROG.write_text(f"SHARD{a.shard} COMPLETE {counts}")
    log.info(f"DONE counts={counts}")

if __name__=="__main__": main()
