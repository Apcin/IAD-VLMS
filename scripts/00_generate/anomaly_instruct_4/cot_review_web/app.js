"use strict";
const $ = id => document.getElementById(id);
const E = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pretty = value => JSON.stringify(value, null, 2);
const filterNames = ["dataset","category","label","status","q","defect_type","visual_judgment","logic_judgment","human","sort","page_size"];
const names = {anomaly_shapenet:"Anomaly ShapeNet",mvtec3d:"MVTec 3D",mvtec_ad:"MVTec AD",real3d:"Real3D"};
const statusNames = {pending:"待评估",done:"已评估",pass:"通过",review:"需复核",fail:"不通过",error:"调用错误"};
const checkNames = {evidence:"图像证据",location:"位置",color:"颜色",shape:"形状",observation_to_inference:"观察 → 推断",inference_to_conclusion:"推断 → 结论",tag_consistency:"标签一致性",uncertainty:"不确定性"};
const verdictNames = {supported:"支持",contradicted:"矛盾",uncertain:"不确定",not_applicable:"不适用"};
const dimensionNames = {visual_faithfulness:"视觉忠实度",reasoning_consistency:"推理合理性"};
const state = {page:1,items:[],total:0,selected:null,data:null,view:0,quote:"",meta:null,listVersion:0,detailVersion:0,drafts:new Map()};
let toastTimer, searchTimer;

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4500);
}
function params() {
  const p = new URLSearchParams();
  for (const name of filterNames) if ($(name).value) p.set(name, $(name).value);
  p.set("page",state.page); return p;
}
function rememberURL() {
  const p = params(); if(state.selected) p.set("sample",state.selected);
  history.replaceState(null,"",`?${p.toString()}${location.hash}`);
}
function options(id, values, label, display = value => value) {
  const previous = $(id).value;
  $(id).innerHTML = `<option value="">${E(label)}</option>` + values.map(v => `<option value="${E(v)}">${E(display(v))}</option>`).join("");
  if (values.includes(previous)) $(id).value = previous;
}
function updateFilterOptions() {
  const selectedDataset = $("dataset").value;
  const rows = state.meta.filters.filter(r => !selectedDataset || r.dataset === selectedDataset);
  options("category",[...new Set(rows.map(r=>r.category))].sort(),"全部类别");
  const category = $("category").value;
  options("defect_type",[...new Set(rows.filter(r=>!category || r.category===category).map(r=>r.defect_type).filter(Boolean))].sort(),"全部类型");
}
async function loadMeta() {
  const meta = await api("/api/meta"); state.meta = meta;
  const judge = meta.summary.judge;
  $("judge-model").textContent = judge?.model || "待选择评估模型";
  $("judge-model").title = judge ? `${judge.provider} · ${judge.endpoint}` : "开始评估后显示当前配置";
  const rows = meta.summary.datasets;
  const sum = field => rows.reduce((n,r)=>n+(r[field]||0),0);
  const total = sum("total"), done = sum("done");
  $("stat-total").textContent = total.toLocaleString();
  $("stat-done").textContent = done.toLocaleString();
  $("stat-review").textContent = (sum("review")+sum("failed")).toLocaleString();
  $("stat-progress").textContent = `${total ? (100*done/total).toFixed(1) : 0}% 已完成 · ${sum("errors")} 条错误`;
  $("progress-bar").style.width = `${total?100*done/total:0}%`;
  $("stat-human").textContent = sum("human_checked").toLocaleString();
  $("stat-flagged").textContent = `待跟进 ${sum("human_flagged").toLocaleString()} 条`;
  options("dataset",rows.map(r=>r.dataset),"全部数据集",v=>names[v]||v);
  $("dataset-count").textContent = `${rows.length} 个数据集 · ${meta.summary.catalog.excluded_failed||0} 条生成失败已排除`;
  $("catalog-info").textContent = rows.map(r=>`${names[r.dataset]||r.dataset}：${r.total.toLocaleString()} 条，已评 ${r.done.toLocaleString()} 条`).join("　/　");
  $("last-updated").textContent = `概况刷新于 ${new Date().toLocaleTimeString("zh-CN")} · 图片与 CoT 直接读取原始文件`;
  updateFilterOptions();
}
function badge(status) { return `<span class="badge ${E(status)}">${E(statusNames[status]||status)}</span>`; }
function judgmentClass(value) { return ["不忠实","不合理"].includes(value)?"negative":value==="不确定"?"uncertain":""; }
function renderList() {
  $("sample-list").innerHTML = state.items.length ? state.items.map(item=>`<button class="sample-card ${item.id===state.selected?"active":""}" data-sample="${item.id}" aria-haspopup="dialog" aria-label="查看 ${E(item.image_id)}">
    <span class="card-picture">${item.thumbnail_url?`<img src="${E(item.thumbnail_url)}" loading="lazy" alt="${E(item.category)} 查询图">`:""}<span class="card-view-count">${item.view_count} 个视角</span></span>
    <span class="card-info"><span class="card-heading"><strong>${E(item.category)}</strong><span class="badge ${item.is_anomaly?"anomaly":"normal"}">${item.is_anomaly?"异常":"正常"}</span></span>
    <span class="card-dataset">${E(names[item.dataset]||item.dataset)}</span><span class="card-id">${E(item.image_id.replace(item.dataset+"__",""))}</span>
    <span class="card-status">${badge(item.verdict||item.status)} ${item.human?`<span class="badge ${E(item.human)}">${item.human==="checked"?"已核查":"待跟进"}</span>`:""}</span>
    <span class="card-judgments">${item.visual_judgment?`<span>视觉 <b class="${judgmentClass(item.visual_judgment)}">${E(item.visual_judgment)}</b></span><span>推理 <b class="${judgmentClass(item.logic_judgment)}">${E(item.logic_judgment)}</b></span>`:item.status==="done"?"历史评分版 · 点击查看":"点击图片查看 CoT 与评估"}</span></span></button>`).join("") : `<div class="empty">没有符合条件的样本。<br>调整筛选条件后重试。</div>`;
  $("sample-list").setAttribute("aria-busy","false");
  $("result-count").textContent = state.total.toLocaleString();
  const pages = Math.max(1,Math.ceil(state.total/Number($("page_size").value)));
  $("page-number").value = state.page; $("page-number").max = pages;
  $("page-total").textContent = `/ ${pages}`;
  $("page-prev").disabled = state.page<=1; $("page-next").disabled = state.page>=pages;
}
async function loadList({keepDetail=true}={}) {
  const version = ++state.listVersion;
  $("sample-list").setAttribute("aria-busy","true");
  try {
    const data = await api(`/api/samples?${params()}`);
    if (version!==state.listVersion) return;
    state.items=data.items; state.total=data.total; state.page=data.page;
    renderList(); rememberURL();
    if(!keepDetail && $("sample-dialog").open && state.selected) await loadDetail(state.selected);
  } catch(error) { $("sample-list").setAttribute("aria-busy","false"); toast(`列表读取失败：${error.message}`); }
}
async function loadDetail(id) {
  const version=++state.detailVersion;
  state.selected=id; state.view=0; state.quote="";
  state.data=null;
  $("detail").innerHTML=`<div class="empty panel">正在读取样本…</div>`;
  if(!$("sample-dialog").open) $("sample-dialog").showModal();
  document.body.classList.add("modal-open");
  $("detail").setAttribute("aria-busy","true");
  renderList(); rememberURL();
  try {
    const data=await api(`/api/sample/${id}`);
    if(version!==state.detailVersion) return;
    state.data=data; renderDetail();
  } catch(error) { if(version===state.detailVersion) $("detail").innerHTML=`<div class="load-error panel">样本读取失败：${E(error.message)}</div>`; }
  finally { if(version===state.detailVersion) $("detail").setAttribute("aria-busy","false"); }
}
function renderDetail() {
  const d=state.data, s=d.sample, evaluation=d.evaluation;
  $("detail").innerHTML=`
    <header class="detail-header panel"><div class="detail-title-line"><h2>${E(s.category)} <span class="muted">/ ${E(names[s.dataset]||s.dataset)}</span></h2>${badge(evaluation?.verdict||(d.error?"error":"pending"))}<div class="detail-nav"><button data-step="-1" title="上一个样本 (K)">← 上一个</button><button data-step="1" title="下一个样本 (J)">下一个 →</button></div></div>
      <div class="detail-id">${E(s.image_id)}</div><div class="detail-meta"><span>${s.is_anomaly?"GT：异常":"GT：正常"}</span><span>类型：${E(s.defect_type||"—")}</span><span>${d.images.query.length} 个查询视角</span><span>${E(d.question)}</span></div></header>
    <section class="panel"><div class="section-bar"><h2>01 <span class="muted">/</span> 图像证据</h2><span class="muted">同一视角 · 点击放大</span></div><div id="view-tabs" class="view-tabs"></div><div id="image-grid" class="image-grid"></div><p id="image-note" class="image-note"></p></section>
    <div class="content-grid"><section class="panel"><div class="section-bar"><h2>02 <span class="muted">/</span> 原始 CoT</h2><button id="copy-cot" class="quiet mono">复制文本</button></div><div id="trace-content" class="trace-content"></div></section>
    <section class="panel"><div class="section-bar"><h2>03 <span class="muted">/</span> 模型评估</h2><span class="muted">两个维度分别判断</span></div><div id="evaluation-content" class="evaluation-body"></div></section></div>
    <section class="panel"><div class="section-bar"><h2>04 <span class="muted">/</span> 人工核查记录</h2><span class="muted">保存在本地 SQLite</span></div><div class="review-body"><label>核查状态<select id="review-decision"><option value="unreviewed">未核查</option><option value="checked">已核查</option><option value="flagged">待跟进</option></select></label><label>备注<textarea id="review-note" maxlength="8000" placeholder="记录需要跟进的细节或你的判断…"></textarea></label><button id="save-review" class="primary">保存记录</button></div></section>`;
  renderGallery(); renderTrace(); renderEvaluation();
  const review=state.drafts.get(s.id)||d.review||{};
  $("review-decision").value=review.decision||"unreviewed"; $("review-note").value=review.note||"";
}
function renderGallery() {
  const d=state.data; if(!d) return;
  state.view=Math.max(0,Math.min(state.view,d.images.query.length-1));
  $("view-tabs").innerHTML=d.images.query.map((q,i)=>`<button data-view="${i}" class="${i===state.view?"active":""}" aria-pressed="${i===state.view}">${E(q.view_id)}</button>`).join("");
  const q=d.images.query[state.view];
  const ref=d.images.reference.find(r=>r.view_id===q.view_id);
  const box=d.bboxes.find(b=>b.view_id===q.view_id);
  const cards=[{title:"原始查询图",item:q,url:q.url},{title:box?"查询图 + GT 框":"查询图 · 无 GT 框",item:q,url:q.bbox_url}, {title:"正常参考图",item:ref,url:ref?.url}];
  $("image-grid").innerHTML=cards.map(card=>`<div class="image-card"><div class="image-label"><span>${E(card.title)}</span><small>${E(q.view_id)} ${q.width}×${q.height}</small></div>${card.url?`<button class="image-button" data-image="${E(card.url)}" data-title="${E(card.title+" / "+q.view_id)}"><img src="${E(card.url)}" alt="${E(card.title)}" loading="lazy"></button><details class="image-path"><summary>原始路径</summary>${E(card.item?.path)}</details>`:`<div class="empty">无此参考图</div>`}</div>`).join("");
  $("image-note").innerHTML=box?`<span class="bbox-label"></span>GT inclusive xyxy：<span class="mono">${E(pretty(box.xyxy_pixel))}</span> · 原图像素坐标；红框用于人工定位，未作为图像输入发送给评估模型。`:`此视角没有 GT 异常框；正常样本保持原图显示。`;
}
function highlighted(text,quote) {
  if (!text || !quote || !text.includes(quote)) return E(text);
  const index=text.indexOf(quote); return E(text.slice(0,index))+`<mark>${E(quote)}</mark>`+E(text.slice(index+quote.length));
}
function renderTrace() {
  const d=state.data,t=d.tags;
  let locationText=t.location;
  try { if(locationText) locationText=pretty(JSON.parse(locationText)); } catch {}
  const quoteInThink=state.quote && t.think?.includes(state.quote);
  $("trace-content").innerHTML=`<div class="tag-title">&lt;think&gt; <small>观察与推理</small></div><p class="think-text">${highlighted(t.think||"未包含 think 标签",state.quote)}</p>
    <div class="tag-block"><div class="tag-title">&lt;location&gt; / &lt;locate&gt;</div>${t.location?`<pre>${E(locationText)}</pre>`:`<span class="muted">未输出（正常样本通常不包含）</span>`}</div>
    <div class="tag-inline"><div><code>&lt;type&gt;</code><b>${E(t.type||"未输出")}</b></div><div><code>&lt;answer&gt;</code><b>${E(t.answer||"未输出")}</b></div></div>
    ${d.trace.quality_warnings?.length?`<details class="source-details"><summary>原生成质量提示 (${d.trace.quality_warnings.length})</summary><pre class="raw-json">${E(pretty(d.trace.quality_warnings))}</pre></details>`:""}
    <details class="source-details" ${state.quote&&!quoteInThink?"open":""}><summary>完整原始结构化文本</summary><pre class="raw-json">${highlighted(d.trace.reasoning,state.quote)}</pre></details>
    <details class="source-details"><summary>原始 evidence_review</summary><pre class="raw-json">${E(pretty(d.trace.evidence_review||[]))}</pre></details>
    <details class="source-details"><summary>原始生成记录与来源</summary><pre class="raw-json">${E(pretty(d.trace))}</pre><p>${E(d.sample.trace_path)}</p><p>${E(d.sample.manifest_path)}</p></details>`;
}
function renderEvaluation() {
  const d=state.data,e=d.evaluation;
  if(!e) { $("evaluation-content").innerHTML=d.error?`<div class="error-state">${E(d.error)}</div><p class="confidence">重新运行评估命令会重试失败样本。</p>`:`<div class="pending-state"><div class="pending-symbol">[ · · · ]</div><h3>等待模型评估</h3><p>可以先对照原图与 CoT 进行人工核查。<br>运行评估脚本后，结果会显示在这里。</p></div>`; return; }
  if(e.evaluation_format==="concise") {
    $("evaluation-content").innerHTML=Object.entries(dimensionNames).map(([key,label])=>{
      const value=e[key],judgment=e.judgments[key],reason=value.replace(/^[^：:]+[：:]\s*/,"");
      return `<section class="judgment-section"><div class="judgment-heading"><h3>${label}</h3><span class="judgment-badge ${judgmentClass(judgment)}">${E(judgment)}</span></div><p>${E(reason)}</p></section>`;
    }).join("")+`<p class="short-evaluation-note">${badge(e.verdict)}　两项正文合计 ${e.text_chars} 字${d.result_updated_at?` · ${E(new Date(d.result_updated_at).toLocaleString("zh-CN"))}`:""}${e.length_limited?"<br>回复已限长，完整原始响应保留在本地评估记录中。":""}</p><details class="source-details"><summary>评估 JSON</summary><pre class="raw-json">${E(pretty(e))}</pre></details>`;
    return;
  }
  $("evaluation-content").innerHTML=`${e.reused_note?`<p class="historical-note">${E(e.reused_note)}</p>`:""}<div class="score-pair">${Object.entries(dimensionNames).map(([key,label])=>`<div class="score-box ${e[key].score<=2?"low":""}"><span>${label}</span><strong>${e[key].score}</strong><small>/ 5</small></div>`).join("")}</div>
    <p class="eval-summary">${E(e.summary)}</p><p class="confidence">${badge(e.verdict)}　评估把握：${E({high:"高",medium:"中",low:"低"}[e.confidence])}${d.result_updated_at?` · ${E(new Date(d.result_updated_at).toLocaleString("zh-CN"))}`:""}</p>
    ${Object.entries(dimensionNames).map(([key,label])=>`<div class="dimension"><h3>${label}</h3><p>${E(e[key].reason)}</p>${Object.entries(e[key].checks).map(([name,c])=>`<div class="check"><span class="check-label">${E(checkNames[name]||name)}</span><span><span class="check-verdict ${E(c.verdict)}">${E(verdictNames[c.verdict])}</span>${E(c.reason)}</span></div>`).join("")}</div>`).join("")}
    <div class="dimension"><h3>问题与证据 <span class="muted">${e.issues.length}</span></h3>${e.issues.length?`<p class="confidence">点击问题定位到原文；有明确视角时会切换图像。</p><ul class="issue-list">${e.issues.map((issue,i)=>`<li class="issue ${state.quote===issue.quote?"selected":""}" data-issue="${i}" tabindex="0" role="button"><div class="issue-top"><span>${E({minor:"轻微",major:"主要",critical:"严重"}[issue.severity])}</span><span>${E(dimensionNames[issue.dimension])}</span><span>${E(issue.view_ids.join(", "))}</span></div><blockquote>${E(issue.quote)}</blockquote><p>${E(issue.explanation)}</p></li>`).join("")}</ul>`:`<p class="confidence">评估模型未列出具体问题。</p>`}</div>
    <details class="source-details"><summary>完整评估 JSON</summary><pre class="raw-json">${E(pretty(e))}</pre></details>`;
}
async function stepSample(direction) {
  if(!$("sample-dialog").open) return;
  const index=state.items.findIndex(item=>item.id===state.selected),next=index+direction;
  if(next>=0 && next<state.items.length) await loadDetail(state.items[next].id);
  else { const pages=Math.max(1,Math.ceil(state.total/Number($("page_size").value))); if(state.page+direction>=1 && state.page+direction<=pages) { state.page+=direction; await loadList(); const item=direction<0?state.items.at(-1):state.items[0]; if(item) await loadDetail(item.id); } }
  $("sample-dialog").scrollTop=0;
}
function chooseIssue(index) {
  const issue=state.data.evaluation.issues[index]; state.quote=issue.quote;
  if(issue.view_ids.length) { const view=state.data.images.query.findIndex(q=>q.view_id===issue.view_ids[0]); if(view>=0) state.view=view; }
  renderGallery(); renderTrace(); renderEvaluation();
}
async function saveReview() {
  const sid=state.selected,payload={decision:$("review-decision").value,note:$("review-note").value};
  if(payload.decision==="unreviewed" && payload.note.trim()) { toast("请先选择“已核查”或“待跟进”，再保存备注。"); return; }
  const button=$("save-review"); button.disabled=true;
  try { await api(`/api/review/${sid}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); state.drafts.delete(sid); if(state.selected===sid && state.data) state.data.review=payload; toast("人工核查记录已保存"); await loadList({keepDetail:true}); }
  catch(error) { toast(`保存失败：${error.message}`); }
  finally { if(button.isConnected) button.disabled=false; }
}
document.addEventListener("click",async event=>{
  const target=event.target.closest("button,[data-issue]"); if(!target) return;
  if(target.dataset.sample) await loadDetail(Number(target.dataset.sample));
  if(target.dataset.view!==undefined) { state.view=Number(target.dataset.view); renderGallery(); }
  if(target.dataset.step) await stepSample(Number(target.dataset.step));
  if(target.dataset.issue!==undefined) chooseIssue(Number(target.dataset.issue));
  if(target.dataset.image) { $("large-image").src=target.dataset.image; $("image-title").textContent=target.dataset.title; $("image-dialog").showModal(); }
  if(target.id==="save-review") await saveReview();
  if(target.id==="copy-cot") { try { await navigator.clipboard.writeText(state.data.trace.reasoning); toast("CoT 已复制"); } catch { toast("复制不可用，请展开完整结构化文本手动复制。"); } }
});
document.addEventListener("input",event=>{ if(["review-note","review-decision"].includes(event.target.id)) state.drafts.set(state.selected,{decision:$("review-decision").value,note:$("review-note").value}); });
document.addEventListener("change",event=>{
  if(event.target.id==="review-decision") state.drafts.set(state.selected,{decision:$("review-decision").value,note:$("review-note").value});
});
document.addEventListener("error",event=>{ if(event.target.tagName==="IMG" && event.target.closest(".image-card")) { const message=document.createElement("p"); message.className="image-error"; message.textContent="图片读取失败，请检查终端日志和原图路径。"; event.target.closest(".image-card").append(message); } },true);
document.addEventListener("keydown",async event=>{
  if(event.key==="Escape") {
    const dialog=$("image-dialog").open?$("image-dialog"):$("sample-dialog").open?$("sample-dialog"):null;
    if(dialog) { event.preventDefault(); event.stopPropagation(); dialog.close(); }
    return;
  }
  if(event.target.matches("input,textarea,select") || $("image-dialog").open || !$("sample-dialog").open) return;
  if(event.target.dataset.issue!==undefined && ["Enter"," "].includes(event.key)) { event.preventDefault(); chooseIssue(Number(event.target.dataset.issue)); return; }
  if(event.key.toLowerCase()==="j") { event.preventDefault(); await stepSample(1); }
  if(event.key.toLowerCase()==="k") { event.preventDefault(); await stepSample(-1); }
  if(["ArrowLeft","ArrowRight"].includes(event.key) && state.data) { event.preventDefault(); state.view+=event.key==="ArrowRight"?1:-1; renderGallery(); }
});
$("close-image").onclick=()=>$("image-dialog").close();
$("close-sample").onclick=()=>$("sample-dialog").close();
$("sample-dialog").addEventListener("close",event=>{
  if(event.target!==$("sample-dialog")) return;
  const previous=state.selected;
  state.detailVersion++; state.selected=null; state.data=null;
  document.body.classList.remove("modal-open"); renderList(); rememberURL();
  document.querySelector(`[data-sample="${previous}"]`)?.focus({preventScroll:true});
});
$("sample-dialog").addEventListener("click",event=>{
  if(event.target!==$("sample-dialog")) return;
  const rect=event.target.getBoundingClientRect();
  if(event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) event.target.close();
});
$("image-dialog").addEventListener("click",event=>{ if(event.target===$("image-dialog")) $("image-dialog").close(); });
for(const name of filterNames) {
  if(name==="q") $(name).addEventListener("input",()=>{ clearTimeout(searchTimer); searchTimer=setTimeout(()=>{state.page=1;loadList();},350); });
  else $(name).addEventListener("change",()=>{ state.page=1; if(["dataset","category"].includes(name)) updateFilterOptions(); loadList(); });
}
$("page-prev").onclick=()=>{state.page--;state.selected=null;loadList();};
$("page-next").onclick=()=>{state.page++;state.selected=null;loadList();};
$("page-number").onchange=()=>{state.page=Number($("page-number").value)||1;state.selected=null;loadList();};
$("reset").onclick=()=>{for(const name of filterNames) $(name).value=name==="sort"?"id":name==="page_size"?"30":"";state.page=1;updateFilterOptions();loadList();};
$("refresh").onclick=async()=>{ try{ await loadMeta(); await loadList({keepDetail:false}); toast("数据已刷新"); }catch(error){toast(error.message);} };
async function start() {
  try {
    const saved=new URLSearchParams(location.search);
    await loadMeta();
    for(const name of ["dataset", "category", ...filterNames.filter(n=>!["dataset","category"].includes(n))]) {
      if(saved.has(name)) $(name).value=saved.get(name);
      if(["dataset","category"].includes(name)) updateFilterOptions();
    }
    if(!$("page_size").value) $("page_size").value="30";
    state.page=Number(saved.get("page"))||1;
    await loadList();
    const savedSample=Number(saved.get("sample"))||null;
    if(savedSample) await loadDetail(savedSample);
    setInterval(async()=>{ try { await loadMeta(); const selected=state.selected; const version=state.detailVersion; if(selected) { const data=await api(`/api/sample/${selected}`); if(selected===state.selected && version===state.detailVersion && state.data) { state.data.evaluation=data.evaluation; state.data.error=data.error;state.data.result_updated_at=data.result_updated_at;renderEvaluation(); } } await loadList({keepDetail:true}); }catch(error){toast(`自动刷新失败：${error.message}`);} },30000);
  } catch(error) { $("sample-list").innerHTML=`<div class="load-error">本地数据读取失败：${E(error.message)}。请检查终端日志后刷新。</div>`; }
}
start();
