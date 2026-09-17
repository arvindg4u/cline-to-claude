"""Native-style status dashboard served at GET / (no cards, no frameworks)."""

from fastapi.responses import HTMLResponse

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cline to Claude Proxy</title>
<style>
:root{
  --bg:#f5f5f7; --panel:#fff; --text:#1d1d1f; --sub:#6e6e73; --hair:rgba(0,0,0,.1);
  --accent:#0071e3; --green:#1d8127; --amber:#b25e09; --red:#c8102e;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',Inter,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#1e1e1e;--panel:#2c2c2e;--text:#f5f5f7;--sub:#a1a1a6;--hair:rgba(255,255,255,.14);
  --accent:#2997ff;--green:#30d158;--amber:#ff9f0a;--red:#ff453a;}
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);font-size:14px;line-height:1.45;
  -webkit-font-smoothing:antialiased}
.window{max-width:900px;margin:0 auto;background:var(--panel);min-height:100dvh;
  border-left:1px solid var(--hair);border-right:1px solid var(--hair)}
.titlebar{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:12px;padding:0 18px;
  height:52px;border-bottom:1px solid var(--hair);background:var(--panel)}
.titlebar h1{font-size:15px;font-weight:600}
.live{margin-left:auto;font-size:12px;color:var(--sub);display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block}
.dot.bad{background:var(--red)}.dot.warn{background:var(--amber)}
main{padding:18px}
h2{font-size:20px;font-weight:700}
.sub{color:var(--sub);font-size:13px;margin-bottom:14px;overflow-wrap:anywhere}
.group{border-top:1px solid var(--hair);margin:18px 0 6px}
.group h3{font-size:12px;font-weight:600;color:var(--sub);text-transform:uppercase;
  letter-spacing:.04em;padding:10px 0 2px}
.row{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--hair)}
.row:last-child{border-bottom:0}
.row .k{flex:none;width:150px;color:var(--sub)}
.row .v{margin-left:auto;text-align:right;overflow-wrap:anywhere;min-width:0;
  font-variant-numeric:tabular-nums}
.row .v.mono{font-family:var(--mono);font-size:12.5px}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:20px;
  background:rgba(128,128,128,.18)}
.badge.ok{color:var(--green)}.badge.warn{color:var(--amber)}.badge.bad{color:var(--red)}
footer{padding:14px 18px;border-top:1px solid var(--hair);color:var(--sub);font-size:12px;
  display:flex;gap:10px}
</style>
</head>
<body>
<div class="window">
  <header class="titlebar">
    <h1>Cline &rarr; Claude Proxy</h1>
    <span class="live"><i id="dot" class="dot"></i><span id="live-label">connecting&hellip;</span></span>
  </header>
  <main>
    <h2 id="ov-status">Starting</h2>
    <p class="sub" id="ov-sub"></p>
    <div class="group"><h3>Proxy</h3><div id="ov-rows"></div></div>
    <div class="group"><h3>Traffic</h3><div id="tr-rows"></div><div id="tr-status"></div></div>
    <div class="group"><h3>Model mapping</h3><div id="mo-map"></div><div id="mo-rows"></div></div>
    <div class="group"><h3>Upstream</h3><div id="up-rows"></div></div>
    <div class="group"><h3>Endpoints</h3><div id="ep-rows"></div></div>
  </main>
  <footer><span id="ft-uptime"></span><span id="ft-refresh"></span></footer>
</div>
<script>
const $=id=>document.getElementById(id);
const row=(k,v,mono)=>'<div class="row"><div class="k">'+k+'</div><div class="v'+(mono?' mono':'')+'">'+v+'</div></div>';
const fmtN=n=>(n||0).toLocaleString();
const fmtT=s=>{s=Math.floor(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return (h?h+'h ':'')+(m?m+'m ':'')+(s%60)+'s';};
const esc=x=>String(x==null?'':x).replace(/</g,'&lt;');
const ago=t=>{const s=Math.max(0,Math.floor(Date.now()/1000-t));
  return s<60?s+'s ago':Math.floor(s/60)+'m ago';};
const dist=(el,m,total)=>el.innerHTML=Object.entries(m||{}).sort((a,b)=>b[1]-a[1])
  .map(([k,v])=>row(esc(k),fmtN(v)+' &middot; '+(total?Math.round(v/total*100):0)+'%')).join('')
  ||'<div class="row"><div class="k">&mdash;</div><div class="v">none</div></div>';
async function tick(){
  let d;
  try{ d=await (await fetch('api/status',{cache:'no-store'})).json(); }
  catch(e){ $('dot').className='dot bad'; $('live-label').textContent='unreachable'; return; }
  const p=d.proxy,s=d.stats;
  const recentErr=s.last_error_at&&((Date.now()/1000)-s.last_error_at)<300;
  const bad=s.error_rate>=0.1&&s.total_requests>0;
  $('dot').className='dot'+(bad?' bad':(recentErr?' warn':''));
  $('live-label').textContent=bad?'degraded':(recentErr?'attention':'live');
  $('ov-status').textContent='Running';
  $('ov-sub').textContent='Cline (OpenAI chat) &rarr; '+p.anthropic_base_url+' (Anthropic messages)';
  $('ov-rows').innerHTML=row('Upstream host',esc(p.anthropic_base_url),1)
    +row('Upstream auth',p.anthropic_auth_mode)
    +row('Client key',p.client_key_validation?'enforced':'open')
    +row('Max tokens limit',fmtN(p.max_tokens_limit))
    +row('Default max tokens',fmtN(p.default_max_tokens))
    +row('Thinking budget',p.thinking_budget_tokens?fmtN(p.thinking_budget_tokens):'disabled')
    +row('Keepalive',p.keepalive_secs+'s')
    +row('Avg latency',s.avg_latency_ms+' ms');
  $('tr-rows').innerHTML=row('Total requests',fmtN(s.total_requests))
    +row('OK / errors',fmtN(s.ok_requests)+' / '+fmtN(s.errors))
    +row('Error rate',(s.error_rate*100).toFixed(1)+'%')
    +row('Tokens in / out',fmtN(s.tokens_in)+' / '+fmtN(s.tokens_out))
    +row('Tokens cached',fmtN(s.tokens_cached)+' ('+(s.tokens_in?Math.round(s.tokens_cached/s.tokens_in*100):0)+'% of in)');
  dist($('tr-status'),s.by_status,s.total_requests);
  const models=p.models||{};
  $('mo-map').innerHTML=row('default',models.default,1)+row('big (opus)',models.big,1)
    +row('middle (sonnet)',models.middle,1)+row('small (haiku)',models.small,1)
    +Object.entries(p.model_map||{}).map(([k,v])=>row('alias '+esc(k),esc(v),1)).join('');
  dist($('mo-rows'),s.by_model,s.total_requests);
  const f=d.last_upstream_failure;
  $('up-rows').innerHTML=(f
      ? row('Last failure',f.status+' &middot; '+ago(f.at))
        +row('error',esc((f.error||'(no error text)').slice(0,300)),1)
        +row('model',esc(f.model||'\u2014'),1)
      : row('Last failure','none recorded'))
    +row('Retries',String(p.max_retries))
    +row('Request timeout',p.request_timeout+'s')
    +(d.failure_history||[]).slice(0,5).map(h=>row(h.status+' &middot; '+ago(h.at),
      esc((h.error||'').slice(0,120)),1)).join('');
  $('ep-rows').innerHTML=['POST /v1/chat/completions|OpenAI wire (translated)',
    'POST /v1/messages|Anthropic passthrough',
    'POST /v1/messages/count_tokens|token counting passthrough',
    'GET /v1/models|model list',
    'GET /health &middot; /test-connection &middot; /api/status|ops']
    .map(x=>{const i=x.indexOf('|');return row(x.slice(0,i),x.slice(i+1),1);}).join('');
  $('ft-uptime').textContent='uptime '+fmtT(s.uptime_secs);
  $('ft-refresh').textContent='refresh '+new Date().toLocaleTimeString();
}
tick();setInterval(()=>{if(!document.hidden)tick();},5000);
</script>
</body>
</html>
"""


def dashboard_response() -> HTMLResponse:
    """Render the status dashboard."""
    return HTMLResponse(content=PAGE)