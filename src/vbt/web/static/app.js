/* The Virtual Biotech web UI: vanilla JS, no build step.
 *
 * Talks to vbt.web.server: JSON endpoints under /api, one Server-Sent-Events
 * stream per session (/api/sessions/<id>/events), downloads under /runs/<id>/.
 * All model and tool text is inserted as text or through esc(); the Markdown
 * renderer escapes before it formats.
 */
'use strict';

(function () {
  const $ = (sel, el) => (el || document).querySelector(sel);
  const $$ = (sel, el) => Array.from((el || document).querySelectorAll(sel));
  const MAX_TOOL_ROWS = 400;
  const STORE_KEY = 'vbt.session';

  const state = {
    config: null, session: null, es: null, lastId: 0, runId: null, busy: false,
    cur: null, agents: new Map(), agentColor: new Map(), tools: new Map(), toolCount: 0, think: new Map(),
    claims: [], refreshTimer: null, turns: 0,
  };

  // ------------------------------------------------------------ utilities

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function fmtUsd(v) {
    if (v == null || isNaN(v)) return '';
    const x = Number(v);
    return x > 0 && x < 1 ? '$' + x.toFixed(3) : '$' + x.toFixed(2);
  }
  function fmtDur(s) {
    if (s == null || isNaN(s)) return '';
    s = Number(s);
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + 's';
    if (s < 3600) return (s / 60).toFixed(1) + ' min';
    return (s / 3600).toFixed(1) + ' h';
  }
  function fmtBytes(n) {
    n = Number(n || 0);
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return i ? n.toFixed(1) + ' ' + u[i] : n + ' B';
  }
  function clock(ts) {
    const d = ts ? new Date(ts * 1000) : new Date();
    return d.toTimeString().slice(0, 8);
  }
  function safeUrl(u) { return /^https?:\/\/[^\s<>"']+$/.test(u) ? u : null; }

  async function api(method, url, body) {
    const opts = { method, headers: {}, credentials: 'same-origin' };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    let res;
    try { res = await fetch(url, opts); } catch (e) { return { status: 0, data: { error: String(e) } }; }
    let data = null;
    try { data = await res.json(); } catch (e) { data = {}; }
    if (res.status === 401 && !url.startsWith('/api/login')) showLogin();
    return { status: res.status, data };
  }

  // ------------------------------------------------------------ markdown (escape first, then format)

  function inline(src, fn) {
    const parts = String(src).split(/(`[^`\n]+`)/);
    return parts.map((p) => {
      if (p.length > 1 && p[0] === '`' && p[p.length - 1] === '`') return '<code>' + esc(p.slice(1, -1)) + '</code>';
      let t = esc(p);
      t = t.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (m, a, b) =>
        '<a href="' + b + '" target="_blank" rel="noopener noreferrer">' + a + '</a>');
      t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
      t = t.replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?!\w)/g, '$1<em>$2</em>');
      t = t.replace(/(^|[^\w])_([^_\s][^_\n]*?)_(?!\w)/g, '$1<em>$2</em>');
      if (fn && fn.size) {
        t = t.replace(/\[([0-9]+(?:†)?|[A-Za-z0-9_.-]+\?)\]/g, (m, label) => {
          const f = fn.get(label);
          if (!f) return m;
          return '<a href="#" class="cref' + (f.missing ? ' missing' : '') + '" data-claim="' + esc(f.id) +
            '" title="' + esc(f.missing ? 'no claim ' + f.id + ' was filed' : f.id + ': ' + (f.text || '')) + '">[' +
            esc(label) + ']</a>';
        });
      }
      return t;
    }).join('');
  }

  function isTableSep(line) { return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line); }
  function cells(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
  }

  function markdown(text, footnotes) {
    const fn = new Map();
    (footnotes || []).forEach((f) => fn.set(String(f.label), f));
    const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let para = [];
    const flush = () => { if (para.length) { out.push('<p>' + inline(para.join('\n'), fn).replace(/\n/g, '<br>') + '</p>'); para = []; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fence = line.match(/^\s*(`{3,}|~{3,})/);
      if (fence) {
        flush();
        const buf = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence[1])) { buf.push(lines[i]); i++; }
        out.push('<pre><code>' + esc(buf.join('\n')) + '</code></pre>');
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { flush(); const n = Math.min(h[1].length + 1, 4); out.push('<h' + n + '>' + inline(h[2], fn) + '</h' + n + '>'); continue; }
      if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { flush(); out.push('<hr>'); continue; }
      if (line.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        flush();
        const head = cells(line);
        let html = '<table><thead><tr>' + head.map((c) => '<th>' + inline(c, fn) + '</th>').join('') + '</tr></thead><tbody>';
        i += 2;
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          html += '<tr>' + cells(lines[i]).map((c) => '<td>' + inline(c, fn) + '</td>').join('') + '</tr>';
          i++;
        }
        i--;
        out.push(html + '</tbody></table>');
        continue;
      }
      if (/^\s*>/.test(line)) {
        flush();
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
        i--;
        out.push('<blockquote>' + markdown(buf.join('\n'), footnotes) + '</blockquote>');
        continue;
      }
      const li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      if (li) {
        flush();
        const ordered = /\d/.test(li[2]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
          if (m) { items.push(m[3]); i++; continue; }
          if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) { items[items.length - 1] += '\n' + lines[i].trim(); i++; continue; }
          break;
        }
        i--;
        const tag = ordered ? 'ol' : 'ul';
        out.push('<' + tag + '>' + items.map((t) => '<li>' + inline(t, fn).replace(/\n/g, '<br>') + '</li>').join('') + '</' + tag + '>');
        continue;
      }
      if (!line.trim()) { flush(); continue; }
      para.push(line);
    }
    flush();
    return out.join('\n');
  }

  // ------------------------------------------------------------ login / boot

  function showLogin() {
    $('#app').hidden = true;
    $('#login').hidden = false;
    if (state.es) { state.es.close(); state.es = null; }
    setTimeout(() => $('#password').focus(), 0);
  }

  async function boot() {
    const r = await api('GET', '/api/auth');
    if (r.status === 0) { document.body.textContent = 'The server is not reachable.'; return; }
    if (r.data.auth_required && !r.data.authenticated) { showLogin(); return; }
    $('#logout').hidden = !r.data.auth_required;
    await start();
  }

  async function login(ev) {
    ev.preventDefault();
    $('#login-error').textContent = '';
    const r = await api('POST', '/api/login', { password: $('#password').value });
    if (r.status === 200) {
      $('#password').value = '';
      $('#login').hidden = true;
      $('#logout').hidden = false;
      await start();
    } else {
      $('#login-error').textContent = r.data.error || 'Sign-in failed';
    }
  }

  async function start() {
    const r = await api('GET', '/api/config');
    if (r.status !== 200) return;
    state.config = r.data;
    $('#login').hidden = true;
    $('#app').hidden = false;
    fillSelectors();
    renderExamples();
    renderAgents();
    let s = null;
    const saved = sessionStorage.getItem(STORE_KEY);
    if (saved) {
      const g = await api('GET', '/api/sessions/' + encodeURIComponent(saved));
      if (g.status === 200) s = g.data;
    }
    if (!s) s = await createSession();
    if (s) attach(s);
    loadRuns();
    $('#prompt').focus();
  }

  // ------------------------------------------------------------ selectors and examples

  function fillSelectors() {
    const c = state.config;
    const prof = $('#profile');
    prof.innerHTML = '';
    const base = (c.base_profiles || []).join(', ') || 'default';
    prof.append(new Option('Server default (' + base + ')', ''));
    (c.profiles || []).forEach((p) => { if (!(c.base_profiles || []).includes(p)) prof.append(new Option(p, p)); });
    const model = $('#model');
    model.innerHTML = '';
    const orch = (c.models || {}).orchestrator || 'configured';
    model.append(new Option('Configured (' + orch + ')', ''));
    const seen = new Set([orch]);
    Object.entries(c.model_aliases || {}).forEach(([alias, id]) => {
      model.append(new Option(alias + (id ? ' (' + id + ')' : ''), alias));
      seen.add(alias);
    });
    Object.values(c.models || {}).forEach((m) => { if (m && !seen.has(m)) { seen.add(m); model.append(new Option(m, m)); } });
  }

  function renderExamples() {
    const box = $('#examples');
    box.innerHTML = '';
    (state.config.examples || []).forEach((ex) => {
      const b = el('button', 'example');
      b.type = 'button';
      b.append(el('span', 't', ex.title), document.createTextNode(ex.prompt));
      b.addEventListener('click', () => { $('#prompt').value = ex.prompt; $('#prompt').focus(); });
      box.append(b);
    });
  }

  // ------------------------------------------------------------ sessions

  async function createSession() {
    const body = { profile: $('#profile').value || null, model: $('#model').value || null };
    const r = await api('POST', '/api/sessions', body);
    if (r.status !== 201) { notice('Could not start a session: ' + (r.data.error || r.status), 'error'); return null; }
    return r.data;
  }

  function resetView() {
    $$('#messages > :not(#welcome)').forEach((n) => n.remove());
    $('#welcome').hidden = false;
    state.cur = null; state.turns = 0; state.toolCount = 0; state.tools.clear(); state.think.clear();
    state.agents.clear(); state.agentColor.clear();
    $('#tools').innerHTML = '';
    $('#tools-empty').hidden = false;
    $('#tool-count').textContent = '';
    renderAgents();
    setRun(null);
    renderClaims([]);
    renderFiles({}, {});
  }

  function attach(s) {
    state.session = s.id;
    sessionStorage.setItem(STORE_KEY, s.id);
    resetView();
    if (s.profile) $('#profile').value = s.profile;
    if (s.model) { const opt = $$('#model option').find((o) => o.value === s.model); if (opt) $('#model').value = s.model; }
    setRun(s.run_id);
    (s.turn_records || []).forEach((t) => {
      addUser(t.prompt);
      const v = newAssistant();
      finishTurn(v, { reply: '', rendered: t.rendered, footnotes: t.footnotes, status: t.status, cost_usd: t.cost_usd,
        claims_filed: t.claims_filed, claims_unresolved: t.claims_unresolved, turn: t.turn });
    });
    setBusy(!!s.busy);
    connect(s.resume_from != null ? s.resume_from : (s.last_event_id || 0));
    if (s.run_id) { refreshClaims(); refreshFiles(); }
  }

  function connect(after) {
    if (state.es) state.es.close();
    state.lastId = after || 0;
    const url = '/api/sessions/' + encodeURIComponent(state.session) + '/events?after=' + state.lastId;
    const es = new EventSource(url);
    state.es = es;
    es.onmessage = (m) => {
      let ev;
      try { ev = JSON.parse(m.data); } catch (e) { return; }
      if (ev.id <= state.lastId) return;
      state.lastId = ev.id;
      handle(ev);
    };
    es.onerror = async () => {
      if (es.readyState === EventSource.CLOSED) {
        const g = await api('GET', '/api/sessions/' + encodeURIComponent(state.session));
        if (g.status === 200) setTimeout(() => { if (state.es === es) connect(state.lastId); }, 1500);
        else if (g.status === 404) { notice('This session has ended. Start a new one to continue.', 'warn'); setBusy(false); }
      } else {
        status('Connection interrupted; reconnecting…');
      }
    };
  }

  // ------------------------------------------------------------ chat rendering

  function scrollDown() { const m = $('#messages'); m.scrollTop = m.scrollHeight; }

  function notice(text, kind) {
    const n = el('div', 'notice' + (kind ? ' ' + kind : ''), text);
    $('#messages').append(n);
    scrollDown();
  }

  function status(text, busy) {
    const s = $('#status-line');
    s.textContent = text || '';
    s.classList.toggle('busy', !!busy);
  }

  function setBusy(b) {
    state.busy = b;
    $('#stop').disabled = !b;
    $('#send').disabled = b;
    $('#new-session').disabled = b;
    if (!b) status('');
  }

  function setRun(runId) {
    state.runId = runId || null;
    const chip = $('#run-chip');
    const zip = $('#dl-run');
    const chat = $('#dl-chat');
    if (runId) {
      chip.textContent = 'Run ' + runId;
      chip.title = runId;
      zip.href = '/runs/' + encodeURIComponent(runId) + '/download.zip';
      chat.href = '/runs/' + encodeURIComponent(runId) + '/chat.md';
      zip.setAttribute('aria-disabled', 'false');
      chat.setAttribute('aria-disabled', 'false');
    } else {
      chip.textContent = 'No run yet: ask a question to start one';
      zip.removeAttribute('href'); chat.removeAttribute('href');
      zip.setAttribute('aria-disabled', 'true');
      chat.setAttribute('aria-disabled', 'true');
    }
  }

  function addUser(text) {
    $('#welcome').hidden = true;
    const m = el('div', 'msg user');
    m.append(el('div', 'bubble', text));
    $('#messages').append(m);
    scrollDown();
  }

  function newAssistant() {
    const m = el('div', 'msg assistant streaming');
    const card = el('div', 'card');
    const who = el('div', 'who', 'CSO');
    const brief = el('details', 'briefing'); brief.hidden = true;
    brief.append(el('summary', null, 'Chief of Staff briefing'), el('div', 'body'));
    const reasoning = el('details', 'reasoning'); reasoning.hidden = true;
    const rsum = el('summary', null, 'Reasoning');
    const rtext = el('div', 'r-text');
    reasoning.append(rsum, rtext);
    const body = el('div', 'body');
    const foot = el('div', 'footnotes'); foot.hidden = true;
    const meta = el('div', 'turn-foot');
    card.append(who, brief, reasoning, body, foot, meta);
    m.append(card);
    $('#messages').append(m);
    const v = { root: m, who, brief, reasoning, rsum, rtext, body, foot, meta, raw: '', thinking: '', pending: false };
    scrollDown();
    return v;
  }

  function renderStream(v) {
    if (v.pending) return;
    v.pending = true;
    requestAnimationFrame(() => {
      v.pending = false;
      if (v.done) return;  // the final rendering (with references) has replaced the stream
      v.body.innerHTML = markdown(v.raw);
      const m = $('#messages');
      if (m.scrollHeight - m.scrollTop - m.clientHeight < 160) scrollDown();
    });
  }

  function finishTurn(v, d) {
    v.done = true;
    v.root.classList.remove('streaming');
    const fns = d.footnotes || [];
    const text = d.rendered != null ? d.rendered : (d.reply || v.raw);
    v.body.innerHTML = markdown(text || v.raw || '', fns);
    if (fns.length) {
      v.foot.hidden = false;
      v.foot.innerHTML = '';
      v.foot.append(el('strong', null, 'References'));
      const ol = el('ol');
      fns.forEach((f) => {
        const li = el('li');
        const lab = el('span', 'fn-l', '[' + f.label + ']');
        li.append(lab);
        if (f.missing) {
          li.append(document.createTextNode(f.id + ': no claim with this id was filed (dangling reference).'));
        } else {
          const a = el('a', null, f.text || f.id);
          a.href = '#';
          a.dataset.claim = f.id;
          li.append(a);
          const parts = (f.evidence || []).map((e) => e.evidence_status || (e.verified ? 'verified' : (e.kind === 'citation' ? 'external' : 'unresolved')));
          const nv = parts.filter((x) => x === 'verified').length;
          li.append(document.createTextNode(' '));
          li.append(el('span', 'vb ' + (nv ? 'verified' : 'unresolved'), nv ? nv + ' verified' : 'no verified evidence'));
        }
        ol.append(li);
      });
      v.foot.append(ol);
    }
    const bits = [];
    if (d.turn != null) bits.push('Turn ' + d.turn);
    if (d.status && d.status !== 'completed') bits.push(String(d.status));
    if (d.cost_usd != null) bits.push(fmtUsd(d.cost_usd));
    if ((d.claims_filed || []).length) bits.push((d.claims_filed || []).length + ' claim(s) filed');
    if ((d.claims_unresolved || []).length) bits.push((d.claims_unresolved || []).length + ' with unresolved evidence');
    v.meta.innerHTML = '';
    bits.forEach((b) => v.meta.append(el('span', null, b)));
    if (state.runId && d.has_audit !== false) {
      const a = el('a', null, 'audit.html');
      a.href = '/runs/' + encodeURIComponent(state.runId) + '/audit.html';
      a.target = '_blank'; a.rel = 'noopener';
      v.meta.append(a);
    }
    if (d.status && d.status !== 'completed') v.who.textContent = 'CSO · ' + d.status;
  }

  // ------------------------------------------------------------ activity sidebar

  function colorClass(agent) {
    if (!agent || agent === 'cso') return 'c0';
    if (!state.agentColor.has(agent)) {
      const n = state.agentColor.size + 1;
      state.agentColor.set(agent, n <= 8 ? 'c' + n : 'c0');
    }
    return state.agentColor.get(agent);
  }

  function renderAgents() {
    const box = $('#agents');
    box.innerHTML = '';
    const roster = (state.config && state.config.roster) || { divisions: {}, cso: { division: 'Office of the CSO' } };
    const divs = Object.assign({}, roster.divisions || {});
    const csoDiv = (roster.cso && roster.cso.division) || 'Office of the CSO';
    divs[csoDiv] = ['cso'].concat((divs[csoDiv] || []).filter((a) => a !== 'cso'));
    const order = [csoDiv].concat(Object.keys(divs).filter((d) => d !== csoDiv));
    order.forEach((d) => {
      const sec = el('div', 'division');
      sec.append(el('h4', null, d));
      const badges = el('div', 'badges');
      divs[d].forEach((name) => {
        const b = el('span', 'badge');
        b.dataset.agent = name;
        const desc = ((roster.agents || []).find((a) => a.name === name) || {}).description || '';
        b.title = desc || name;
        b.append(el('span', 'dot'), el('span', null, name === 'cso' ? 'CSO' : name), el('span', 'n'));
        badges.append(b);
      });
      sec.append(badges);
      box.append(sec);
    });
    state.agents.forEach((st, name) => paintAgent(name, st));
  }

  function agentState(name) {
    if (!state.agents.has(name)) state.agents.set(name, { running: 0, runs: 0, errors: 0 });
    return state.agents.get(name);
  }

  function paintAgent(name, st) {
    let b = $('.badge[data-agent="' + CSS.escape(name) + '"]');
    if (!b) {
      let other = $('.division[data-other] .badges');
      if (!other) {
        const sec = el('div', 'division');
        sec.dataset.other = '1';
        sec.append(el('h4', null, 'Other'));
        other = el('div', 'badges');
        sec.append(other);
        $('#agents').append(sec);
      }
      b = el('span', 'badge');
      b.dataset.agent = name;
      b.append(el('span', 'dot'), el('span', null, name), el('span', 'n'));
      other.append(b);
    }
    b.classList.toggle('running', st.running > 0);
    b.classList.toggle('done', st.running === 0 && st.runs > 0 && !st.errors);
    b.classList.toggle('error', st.running === 0 && st.errors > 0);
    $('.n', b).textContent = st.runs > 1 ? '×' + st.runs : '';
  }

  function toolRow(id, d, ts) {
    $('#tools-empty').hidden = true;
    const li = el('li');
    const row = el('div', 'row');
    row.append(el('span', 't', clock(ts)), el('span', 'ag ' + colorClass(d.agent), d.agent || '?'),
      el('span', 'tn', d.tool || '?'), el('span', 'pill running', 'running'));
    li.append(row);
    if (d.input_preview) li.title = d.input_preview;
    const list = $('#tools');
    list.prepend(li);
    state.toolCount++;
    while (list.children.length > MAX_TOOL_ROWS) {
      const last = list.lastElementChild;
      if (last.dataset.id) state.tools.delete(last.dataset.id);
      last.remove();
    }
    $('#tool-count').textContent = state.toolCount > MAX_TOOL_ROWS ?
      '(' + state.toolCount + '; latest ' + MAX_TOOL_ROWS + ' shown, all in audit.html)' : '(' + state.toolCount + ')';
    if (id) { li.dataset.id = id; state.tools.set(id, { li, agent: d.agent, legacy: !!d.legacy }); }
    return li;
  }

  function infoRow(text, ts, kind) {
    $('#tools-empty').hidden = true;
    const li = el('li', 'info');
    const row = el('div', 'row');
    row.append(el('span', 't', clock(ts)), el('span', 'tn', text));
    if (kind) row.append(el('span', 'pill ' + kind, kind === 'error' ? 'error' : kind === 'ok' ? 'done' : kind));
    li.append(row);
    $('#tools').prepend(li);
    return li;
  }

  function setPill(li, cls, text) {
    const p = $('.pill', li);
    if (p) { p.className = 'pill ' + cls; p.textContent = text; }
  }

  // ------------------------------------------------------------ event handling

  function handle(ev) {
    const d = ev.data || {};
    switch (ev.kind) {
      case 'user':
        state.turnSeq = (state.turnSeq || 0) + 1;
        addUser(d.prompt);
        state.cur = newAssistant();
        setBusy(true);
        status('Working…', true);
        break;
      case 'status':
        if (state.busy) status(d.message, true);
        break;
      case 'session_started':
        setRun(d.run_id);
        break;
      case 'turn_start':
        if (state.cur && d.turn != null) state.cur.who.textContent = 'CSO · turn ' + d.turn;
        break;
      case 'text':
        if (d.agent === 'cso') {
          if (!state.cur) state.cur = newAssistant();
          state.cur.raw += d.text || '';
          renderStream(state.cur);
        }
        break;
      case 'message_end':
        if (d.agent === 'cso' && state.cur && state.cur.raw && !/\n\n$/.test(state.cur.raw)) state.cur.raw += '\n\n';
        { const t = state.think.get(thinkKey(d)); if (t) t.sep = true; }
        break;
      case 'thinking': {
        // Streamed deltas are concatenated; a complete block starts a new paragraph
        // (and is skipped when its deltas were already shown).
        const key = thinkKey(d);
        let t = state.think.get(key);
        if (!t) {
          t = { text: '', sep: false, view: null };
          state.think.set(key, t);
          if (d.agent === 'cso' && state.cur) {
            t.view = { box: state.cur.reasoning, text: state.cur.rtext, sum: state.cur.rsum, cso: true };
          } else {
            const li = infoRow((d.agent || '?') + ' reasoning', ev.ts);
            const det = el('details');
            const out = el('div', 'out');
            const sum = el('summary', null, 'show');
            det.append(sum, out);
            li.append(det);
            t.view = { box: null, text: out, sum: sum, cso: false };
          }
        }
        const chunk = String(d.text || '');
        if (!chunk) break;
        if (d.streamed) {
          if (t.sep && t.text) t.text += '\n\n';
          t.text += chunk;
        } else if (!t.text.endsWith(chunk.trim())) {
          t.text += (t.text ? '\n\n' : '') + chunk;
        }
        t.sep = false;
        if (t.view.box) t.view.box.hidden = false;
        t.view.text.textContent = t.view.cso ? t.text : t.text.slice(-6000);
        t.view.sum.textContent = (t.view.cso ? 'Reasoning' : 'show') + ' (' + t.text.length.toLocaleString() + ' chars)';
        break;
      }
      case 'briefing':
        if (state.cur) {
          state.cur.brief.hidden = false;
          $('.body', state.cur.brief).innerHTML = markdown(d.text || '');
        }
        break;
      case 'delegation': {
        const st = agentState(d.agent);
        paintAgent(d.agent, st);
        infoRow('→ ' + d.agent + (d.description ? ': ' + d.description : ''), ev.ts);
        status('Delegated to ' + d.agent, true);
        break;
      }
      case 'agent_start':
        if ((d.depth || 0) >= 1 || d.agent === 'cso') {
          const st = agentState(d.agent);
          st.running++; st.runs++;
          paintAgent(d.agent, st);
        }
        break;
      case 'agent_end': {
        const st = agentState(d.agent);
        st.running = Math.max(0, st.running - 1);
        if (d.status && !['completed', 'end_turn'].includes(d.status)) st.errors++;
        paintAgent(d.agent, st);
        if ((d.depth || 0) >= 1) {
          infoRow('✓ ' + d.agent + ' finished' + (d.cost_usd != null ? ' (' + fmtUsd(d.cost_usd) + ')' : '') +
            (d.status && d.status !== 'completed' ? ' · ' + d.status : ''), ev.ts,
            d.status && d.status !== 'completed' ? 'error' : null);
        }
        state.tools.forEach((t, id) => {
          if (t.legacy && t.agent === d.agent) { setPill(t.li, 'done', 'done'); state.tools.delete(id); }
        });
        break;
      }
      case 'tool_start':
        toolRow(d.tool_use_id, d, ev.ts);
        if (state.busy) status((d.agent || '?') + ': ' + (d.tool || ''), true);
        break;
      case 'tool_end': {
        const t = state.tools.get(d.tool_use_id);
        const li = t ? t.li : toolRow(d.tool_use_id, d, ev.ts);
        setPill(li, d.is_error ? 'error' : 'ok', d.is_error ? 'error' : 'ok');
        if (d.duration_s != null) {
          let meta = $('.meta', li);
          if (!meta) { meta = el('div', 'meta'); li.append(meta); }
          meta.textContent = fmtDur(d.duration_s);
        }
        if (d.is_error && d.output_preview) {
          const det = el('details');
          det.append(el('summary', null, 'error output'), el('div', 'out', d.output_preview));
          li.append(det);
        }
        state.tools.delete(d.tool_use_id);
        break;
      }
      case 'draft_superseded':
        notice('The text above is a draft, superseded' + (d.reason ? ' (' + d.reason + ' requested)' : '') + '.');
        break;
      case 'review_enforced':
        notice('Review required: dispatching the scientific reviewer' + (d.agents ? ' for ' + [].concat(d.agents).join(', ') : '') + '.');
        break;
      case 'warning':
        infoRow('Warning: ' + (d.message || ''), ev.ts, 'error');
        break;
      case 'retry':
        infoRow('Retrying a model call' + (d.delay_s != null ? ' in ' + fmtDur(d.delay_s) : '') + (d.error ? ': ' + d.error : ''), ev.ts);
        break;
      case 'compaction':
        infoRow('Context compacted for ' + (d.agent || '?') + (d.strategy ? ' (' + d.strategy + ')' : ''), ev.ts);
        break;
      case 'mcp_crash': case 'mcp_timeout': case 'mcp_start_failed':
        infoRow(ev.kind.replace('mcp_', 'MCP ').replace('_', ' ') + ': ' + (d.server || d.name || '') + (d.error ? ' ' + d.error : ''), ev.ts, 'error');
        break;
      case 'artifact_registered': case 'claims_filed':
        scheduleRefresh();
        break;
      case 'error':
        notice(d.message || 'Error', d.message === 'previous query still processing' ? 'warn' : 'error');
        break;
      case 'turn_end': {
        if (!state.cur) state.cur = newAssistant();
        finishTurn(state.cur, d);
        state.cur = null;
        state.turns++;
        state.tools.forEach((t) => setPill(t.li, 'done', 'done'));
        state.tools.clear();
        state.think.clear();
        state.agents.forEach((st, name) => { st.running = 0; paintAgent(name, st); });
        setBusy(false);
        if (d.run_id) setRun(d.run_id);
        refreshClaims(); refreshFiles(); loadRuns();
        break;
      }
      case 'session_closed':
        notice('Session closed. Its record has been finalised.', 'warn');
        setBusy(false);
        break;
      default:
        break;
    }
  }

  function thinkKey(d) {
    // CSO reasoning belongs to the current turn's card; specialists' to their invocation.
    return d.agent === 'cso' ? 'cso|' + (state.turnSeq || 0) : (d.agent || '?') + '|' + (d.invocation_id || '');
  }

  function scheduleRefresh() {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = setTimeout(() => { refreshClaims(); refreshFiles(); }, 800);
  }

  // ------------------------------------------------------------ evidence panel

  async function refreshClaims() {
    if (!state.session) return;
    const r = await api('GET', '/api/sessions/' + encodeURIComponent(state.session) + '/claims');
    if (r.status === 200) renderClaims(r.data.claims || [], r.data.stats);
  }

  function renderClaims(claims, stats) {
    state.claims = claims;
    const box = $('#claims');
    box.innerHTML = '';
    $('#claims-count').textContent = claims.length ? '(' + claims.length + ')' : '';
    if (!claims.length) { $('#claims-stats').textContent = 'No claims filed yet.'; return; }
    if (stats) {
      $('#claims-stats').textContent = claims.length + ' claim(s): ' + (stats.n_verified_evidence || 0) + ' verified, ' +
        (stats.n_external_evidence || 0) + ' external, ' + (stats.n_unresolved_evidence || 0) + ' unresolved evidence item(s)';
    }
    const onlyBad = $('#claims-unverified').checked;
    claims.forEach((c) => {
      if (onlyBad && c.n_verified) return;
      const card = el('div', 'claim' + (c.n_verified ? '' : ' warn'));
      card.id = 'claim-' + c.id;
      const head = el('div', 'ch');
      head.append(el('span', 'cid', c.id), el('span', 'ct', c.text));
      card.append(head);
      const meta = [c.agent || 'unattributed', 'confidence ' + (c.confidence || 'moderate')];
      if (c.turn != null) meta.push('turn ' + c.turn);
      card.append(el('div', 'cm', meta.join(' · ')));
      const ul = el('ul');
      (c.evidence || []).forEach((e) => {
        const li = el('li');
        if (e.url) {
          const a = el('a', null, e.text);
          a.href = e.url; a.target = '_blank'; a.rel = 'noopener';
          li.append(a);
        } else {
          li.append(document.createTextNode(e.text));
        }
        if (e.note) li.append(document.createTextNode(' — ' + e.note));
        li.append(el('span', 'vb ' + e.status, e.label));
        ul.append(li);
      });
      card.append(ul);
      box.append(card);
    });
  }

  function showClaim(id) {
    selectTab('evidence');
    const card = document.getElementById('claim-' + id);
    if (card) {
      card.scrollIntoView({ block: 'center', behavior: 'smooth' });
      card.classList.add('flash');
      setTimeout(() => card.classList.remove('flash'), 1600);
    }
  }

  // ------------------------------------------------------------ files panel

  async function refreshFiles() {
    if (!state.session) return;
    const r = await api('GET', '/api/sessions/' + encodeURIComponent(state.session) + '/files');
    if (r.status === 200) renderFiles(r.data.files || {}, r.data.counts || {});
  }

  function renderFiles(files, counts) {
    const gallery = $('#gallery');
    const list = $('#filelist');
    gallery.innerHTML = ''; list.innerHTML = '';
    let total = 0;
    Object.values(counts || {}).forEach((n) => { total += n; });
    $('#files-count').textContent = total ? '(' + total + ')' : '';
    $('#files-stats').textContent = total ? total + ' file(s) under work/' : 'No files yet.';
    Object.entries(files.figures || {}).forEach(([agent, rows]) => rows.forEach((r) => {
      if (!r.image) return;
      const a = el('a');
      a.href = r.url; a.target = '_blank'; a.rel = 'noopener'; a.title = r.path;
      const img = el('img');
      img.loading = 'lazy'; img.alt = r.name; img.src = r.url;
      a.append(img, el('div', 'cap', agent + ' · ' + r.name));
      gallery.append(a);
    }));
    [['figures', 'Figures'], ['tables', 'Tables'], ['code', 'Code'], ['reports', 'Reports'], ['data', 'Data'], ['other', 'Other']]
      .forEach(([key, label]) => {
        const groups = files[key] || {};
        const n = Object.values(groups).reduce((s, rows) => s + rows.length, 0);
        if (!n) return;
        const sec = el('div', 'filegroup');
        sec.append(el('h4', null, label + ' (' + n + ')'));
        Object.entries(groups).forEach(([agent, rows]) => {
          const ul = el('ul');
          rows.forEach((r) => {
            const li = el('li');
            const a = el('a', null, r.path.replace(/^work\//, ''));
            a.href = r.url; a.target = '_blank'; a.rel = 'noopener'; a.title = r.path + (r.description ? '\n' + r.description : '');
            li.append(a, el('span', 'sz', fmtBytes(r.bytes)));
            ul.append(li);
          });
          sec.append(el('div', 'muted small', agent), ul);
        });
        list.append(sec);
      });
  }

  // ------------------------------------------------------------ past runs

  async function loadRuns() {
    const r = await api('GET', '/api/runs?limit=100');
    if (r.status !== 200) return;
    const list = $('#runs-list');
    list.innerHTML = '';
    const runs = r.data.runs || [];
    $('#runs-stats').textContent = runs.length ? runs.length + ' run(s), newest first' : 'No runs yet.';
    runs.forEach((run) => {
      const li = el('li');
      const q = el('div', 'q', run.query || '(no query recorded)');
      q.title = run.query || '';
      const meta = el('div', 'meta');
      meta.append(el('span', 'st ' + String(run.status || '').replace(/[^a-z_]/g, ''), String(run.status || '?').replace(/_/g, ' ')));
      const id = el('span', 'id', run.run_id);
      meta.append(id);
      if (run.created) meta.append(el('span', null, String(run.created).slice(0, 16).replace('T', ' ')));
      meta.append(el('span', null, (run.n_turns || 0) + ' turn(s)'));
      meta.append(el('span', null, (run.n_claims || 0) + ' claim(s)' + (run.n_unresolved_evidence ? ', ' + run.n_unresolved_evidence + ' unresolved' : '')));
      if (run.cost_usd != null) meta.append(el('span', null, fmtUsd(run.cost_usd)));
      const links = el('div', 'links');
      if (run.audit_url) { const a = el('a', null, 'audit.html'); a.href = run.audit_url; a.target = '_blank'; a.rel = 'noopener'; links.append(a); }
      const z = el('a', null, 'run (.zip)'); z.href = run.zip_url; links.append(z);
      const c = el('a', null, 'chat (.md)'); c.href = run.chat_url; links.append(c);
      li.append(q, meta, links);
      list.append(li);
    });
  }

  // ------------------------------------------------------------ tabs and controls

  function selectTab(name) {
    $$('.tabs [role="tab"]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    $$('.panel').forEach((p) => { p.hidden = p.id !== 'panel-' + name; });
    if (name === 'files') refreshFiles();
    if (name === 'evidence') refreshClaims();
    if (name === 'runs') loadRuns();
  }

  async function send(ev) {
    if (ev) ev.preventDefault();
    const text = $('#prompt').value.trim();
    if (!text || !state.session) return;
    if (state.busy) { notice('The previous query is still processing. Wait for it to finish or press Stop.', 'warn'); return; }
    $('#send').disabled = true;
    const r = await api('POST', '/api/sessions/' + encodeURIComponent(state.session) + '/ask', { prompt: text });
    if (r.status === 202) {
      $('#prompt').value = '';
    } else if (r.status === 409) {
      notice('The previous query is still processing. Wait for it to finish or press Stop.', 'warn');
    } else if (r.status === 404) {
      notice('This session has ended; starting a new one.', 'warn');
      const s = await createSession();
      if (s) attach(s);
    } else {
      notice('Could not send: ' + (r.data.error || r.status), 'error');
      $('#send').disabled = state.busy;
    }
  }

  async function stop() {
    if (!state.session) return;
    $('#stop').disabled = true;
    status('Stopping…', true);
    const r = await api('POST', '/api/sessions/' + encodeURIComponent(state.session) + '/stop', {});
    if (r.status !== 200 || !r.data.ok) status(r.data.detail || r.data.error || 'Nothing to stop');
  }

  async function newSession() {
    if (state.busy) return;
    if (state.session && state.turns > 0 &&
        !window.confirm('Close this session (its run record is finalised) and start a new one?')) return;
    if (state.session) await api('DELETE', '/api/sessions/' + encodeURIComponent(state.session));
    const s = await createSession();
    if (s) attach(s);
  }

  async function optionsChanged() {
    if (!state.busy && state.turns === 0 && !state.runId) {
      if (state.session) await api('DELETE', '/api/sessions/' + encodeURIComponent(state.session));
      const s = await createSession();
      if (s) attach(s);
    } else {
      status('The new profile/model applies to the next session (New session).');
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('#login-form').addEventListener('submit', login);
    $('#composer').addEventListener('submit', send);
    $('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) send(e); });
    $('#stop').addEventListener('click', stop);
    $('#new-session').addEventListener('click', newSession);
    $('#profile').addEventListener('change', optionsChanged);
    $('#model').addEventListener('change', optionsChanged);
    $('#logout').addEventListener('click', async () => { await api('POST', '/api/logout', {}); sessionStorage.removeItem(STORE_KEY); showLogin(); });
    $('#files-refresh').addEventListener('click', refreshFiles);
    $('#runs-refresh').addEventListener('click', loadRuns);
    $('#claims-unverified').addEventListener('change', () => renderClaims(state.claims));
    $$('.tabs [role="tab"]').forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a[data-claim]');
      if (a) { e.preventDefault(); showClaim(a.dataset.claim); }
      const dl = e.target.closest('a[aria-disabled="true"]');
      if (dl) e.preventDefault();
    });
    boot();
  });
})();
