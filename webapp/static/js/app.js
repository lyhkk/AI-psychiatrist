'use strict';

/* ── 状态 ──────────────────────────────────────────────────── */
const State = {
  sessionId: null,
  isWaiting: false,   // 是否正在等待 AI 回复
  panelVisible: true,
};

/* ── DOM 引用 ───────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const el = {
  welcomeScreen : $('welcome-screen'),
  welcomeInput  : $('welcome-input'),
  btnStart      : $('btn-welcome-start'),
  messages      : $('messages'),
  inputBar      : $('input-bar'),
  userInput     : $('user-input'),
  btnSend       : $('btn-send'),
  btnNewSession : $('btn-new-session'),
  btnTogglePanel: $('btn-toggle-panel'),
  panel         : $('cbt-panel'),
  toast         : $('toast'),
  valSituation  : $('val-situation'),
  valEmotion    : $('val-emotion'),
  valThought    : $('val-thought'),
  valDistortion : $('val-distortion'),
  cardSituation : $('card-situation'),
  cardEmotion   : $('card-emotion'),
  cardThought   : $('card-thought'),
  cardDistortion: $('card-distortion'),
  btnHistory    : $('btn-history'),
  drawer        : $('history-drawer'),
  drawerBackdrop: $('drawer-backdrop'),
  drawerList    : $('drawer-list'),
  btnCloseHistory: $('btn-close-history'),
  btnExportJson : $('btn-export-json'),
  btnExportMd   : $('btn-export-md'),
  btnUndo       : $('btn-undo'),
  btnTracker    : $('btn-tracker'),
  trackerView   : $('tracker-view'),
  trackerBody   : $('tracker-body'),
  btnTrackerBack: $('btn-tracker-back'),
};

/* ── 工具函数 ───────────────────────────────────────────────── */
function showToast(msg, duration = 3000) {
  el.toast.textContent = msg;
  el.toast.classList.add('show');
  setTimeout(() => el.toast.classList.remove('show'), duration);
}

function formatTime(isoStr) {
  const d = isoStr ? new Date(isoStr) : new Date();
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

/* ── 渲染：追加用户消息 ──────────────────────────────────────── */
function appendUserMsg(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap msg-wrap-user';
  const row = document.createElement('div');
  row.className = 'msg msg-user';
  row.innerHTML = `
    <div class="msg-avatar">&#128100;</div>
    <div class="msg-bubble">${escHtml(text)}</div>`;
  const time = document.createElement('p');
  time.className = 'msg-time';
  time.textContent = formatTime();
  wrap.appendChild(row);
  wrap.appendChild(time);
  el.messages.appendChild(wrap);
  scrollToBottom();
}

/* ── 渲染：追加 AI 消息（一次性） ───────────────────────────── */
function appendAiMsg(text, isoTime) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap msg-wrap-ai';
  const row = document.createElement('div');
  row.className = 'msg msg-ai';
  row.innerHTML = '<div class="msg-avatar">&#9775;</div><div class="msg-bubble"></div>';
  row.querySelector('.msg-bubble').textContent = text;
  const time = document.createElement('p');
  time.className = 'msg-time';
  time.textContent = formatTime(isoTime);
  wrap.appendChild(row);
  wrap.appendChild(time);
  el.messages.appendChild(wrap);
  scrollToBottom();
}

/* ── 渲染：思考指示器 ────────────────────────────────────────── */
function showThinking() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap msg-wrap-ai';
  wrap.id = 'thinking-wrap';
  wrap.innerHTML = '<div class="msg msg-ai thinking"><div class="msg-avatar">&#9775;</div><div class="msg-bubble"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span></div></div>';
  el.messages.appendChild(wrap);
  scrollToBottom();
}
function hideThinking() {
  const t = document.getElementById('thinking-wrap');
  if (t) t.remove();
}

/* ── HTML 转义 ───────────────────────────────────────────────── */
function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── CBT 面板更新 ────────────────────────────────────────────── */
const FIELD_MAP = [
  { key: 'situation',          el: () => el.valSituation,  card: () => el.cardSituation  },
  { key: 'emotion',            el: () => el.valEmotion,    card: () => el.cardEmotion    },
  { key: 'automatic_thought',  el: () => el.valThought,    card: () => el.cardThought    },
  { key: 'cognitive_distortion', el: () => el.valDistortion, card: () => el.cardDistortion },
];

function updateCbtPanel(form) {
  if (!form) return;
  FIELD_MAP.forEach(({ key, el: getEl, card: getCard }) => {
    const val = form[key];
    const valEl  = getEl();
    const cardEl = getCard();
    const oldText = valEl.textContent;
    if (val && val !== 'null' && val !== 'None') {
      valEl.textContent = val;
      valEl.className = 'cbt-card-value has-value';
    } else {
      valEl.textContent = '暂未识别';
      valEl.className = 'cbt-card-value null-value';
    }
    // flash animation when value changes
    if (valEl.textContent !== oldText) {
      cardEl.classList.remove('updated');
      void cardEl.offsetWidth; // reflow
      cardEl.classList.add('updated');
      setTimeout(() => cardEl.classList.remove('updated'), 700);
    }
  });
}

/* ── 会话管理 ────────────────────────────────────────────────── */
async function startSession(opening = '') {
  try {
    const res = await fetch('/api/chat/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ opening }),
    });
    const data = await res.json();
    if (data.status !== 'ok') throw new Error(data.message);
    State.sessionId = data.session_id;
    return true;
  } catch(e) {
    showToast('创建会话失败：' + e.message);
    return false;
  }
}

/* ── 显示对话界面 ────────────────────────────────────────────── */
function showChatUI() {
  el.welcomeScreen.style.display = 'none';
  el.messages.style.display = 'flex';
  el.inputBar.style.display = 'block';
  el.userInput.focus();
}

/* ── 发送消息（普通 fetch，等待完整响应） ───────────────────── */
async function sendMessage(text) {
  if (State.isWaiting || !text.trim()) return;
  if (!State.sessionId) {
    showToast('请先开始对话');
    return;
  }

  State.isWaiting = true;
  el.btnSend.disabled = true;
  el.userInput.disabled = true;

  appendUserMsg(text);
  el.userInput.value = '';
  el.userInput.style.height = 'auto';

  showThinking();

  try {
    const res = await fetch('/api/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();
    hideThinking();

    if (data.status !== 'ok') {
      showToast('错误：' + (data.message || '未知错误'));
      return;
    }

    appendAiMsg(data.reply, data.timestamp);
    updateCbtPanel(data.cbt_form);

  } catch(e) {
    hideThinking();
    showToast('发送失败：' + e.message);
  } finally {
    State.isWaiting = false;
    el.btnSend.disabled = false;
    el.userInput.disabled = false;
    el.userInput.focus();
  }
}

/* ── 新对话 ──────────────────────────────────────────────────── */
async function newSession() {
  if (State.isWaiting) return;
  // 结束旧会话
  if (State.sessionId) {
    fetch('/api/chat/session', { method: 'DELETE' }).catch(() => {});
    State.sessionId = null;
  }
  // 清空消息
  el.messages.innerHTML = '';
  // 重置 CBT 面板
  FIELD_MAP.forEach(({ el: getEl }) => {
    getEl().textContent = '暂未识别';
    getEl().className = 'cbt-card-value null-value';
  });
  // 重新显示欢迎屏
  el.welcomeScreen.style.display = 'flex';
  el.welcomeInput.value = '';
  el.messages.style.display = 'none';
  el.inputBar.style.display = 'none';
  el.welcomeInput.focus();
  showToast('已开启新对话');
}

/* ── 会话历史抽屉 ────────────────────────────────────────────── */
function openHistory() {
  el.drawer.classList.add('show');
  el.drawerBackdrop.classList.add('show');
  el.drawer.setAttribute('aria-hidden', 'false');
  loadHistoryList();
}
function closeHistory() {
  el.drawer.classList.remove('show');
  el.drawerBackdrop.classList.remove('show');
  el.drawer.setAttribute('aria-hidden', 'true');
}
async function loadHistoryList() {
  el.drawerList.innerHTML = '<p class="hist-empty">加载中……</p>';
  try {
    const res = await fetch('/api/chat/sessions');
    const data = await res.json();
    if (data.status !== 'ok') throw new Error(data.message || '加载失败');
    renderHistoryList(data.sessions || []);
  } catch (e) {
    el.drawerList.innerHTML = '<p class="hist-empty">加载失败：' + escHtml(e.message) + '</p>';
  }
}
function renderHistoryList(sessions) {
  if (!sessions.length) {
    el.drawerList.innerHTML = '<p class="hist-empty">还没有历史会话</p>';
    return;
  }
  el.drawerList.innerHTML = '';
  sessions.forEach(s => {
    const row = document.createElement('div');
    row.className = 'hist-row';
    const badge = s.cognitive_distortion
      ? `<span class="hist-badge">${escHtml(s.cognitive_distortion)}</span>` : '';
    const current = s.session_id === State.sessionId
      ? '<span class="hist-badge">当前</span>' : '';
    row.innerHTML = `
      <div class="hist-title">${escHtml(s.title || '(空会话)')}</div>
      <div class="hist-meta">
        <span>${formatDate(s.last_active)}</span>
        <span>· ${s.turn_count} 轮</span>
        ${badge}${current}
      </div>`;
    const actions = document.createElement('div');
    actions.className = 'hist-actions';
    actions.appendChild(mkBtn('继续', () => resumeSession(s.session_id)));
    actions.appendChild(mkBtn('JSON', () => exportSession(s.session_id, 'json')));
    actions.appendChild(mkBtn('MD',   () => exportSession(s.session_id, 'md')));
    row.appendChild(actions);
    el.drawerList.appendChild(row);
  });
}
function mkBtn(label, fn) {
  const b = document.createElement('button');
  b.className = 'btn btn-ghost btn-sm';
  b.textContent = label;
  b.addEventListener('click', fn);
  return b;
}
function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '—';
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

/* ── 渲染一段历史消息（复用现有气泡组件，保证视觉一致） ──────── */
function renderHistoryMessages(history) {
  el.messages.innerHTML = '';
  (history || []).forEach(m => {
    if (m.role === 'user') appendUserMsg(m.content);
    else if (m.role === 'assistant') appendAiMsg(m.content);
  });
  scrollToBottom();
}

/* ── 继续某段历史会话 ────────────────────────────────────────── */
async function resumeSession(sid) {
  if (State.isWaiting) return;
  try {
    const res = await fetch('/api/chat/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid }),
    });
    const data = await res.json();
    if (data.status !== 'ok') throw new Error(data.message || '恢复失败');
    State.sessionId = data.session_id;
    closeHistory();
    showChatUI();
    renderHistoryMessages(data.history);
    updateCbtPanel(data.cbt_form);
    showToast('已恢复会话 · 接着上次继续');
  } catch (e) {
    showToast('恢复失败：' + e.message);
  }
}

/* ── 下载会话（不离开页面） ──────────────────────────────────── */
function exportSession(sid, fmt) {
  if (!sid) { showToast('请先开始或选择一个会话'); return; }
  const a = document.createElement('a');
  a.href = `/api/chat/sessions/${encodeURIComponent(sid)}/export?format=${fmt}`;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast(`正在下载 ${fmt.toUpperCase()}`);
}

/* ── 撤销上一轮 ──────────────────────────────────────────────── */
async function undoLastTurn() {
  if (State.isWaiting) return;
  if (!State.sessionId) { showToast('请先开始或选择一个会话'); return; }
  try {
    const res = await fetch('/api/chat/undo', { method: 'POST' });
    const data = await res.json();
    if (data.status !== 'ok') { showToast(data.message || '撤销失败'); return; }
    renderHistoryMessages(data.history);
    updateCbtPanel(data.cbt_form);
    showToast('已撤销上一轮');
  } catch (e) {
    showToast('撤销失败：' + e.message);
  }
}

/* ── 面板开关 ────────────────────────────────────────────────── */
function togglePanel() {
  State.panelVisible = !State.panelVisible;
  el.panel.classList.toggle('hidden', !State.panelVisible);
}

/* ── 自适应文本框高度 ────────────────────────────────────────── */
function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
}

/* ── 追踪：视图切换 ───────────────────────────────────────────── */
const Tracker = { baselineId: null, checkinId: null };
function showTracker() {
  document.querySelector('.layout').style.display = 'none';
  el.trackerView.style.display = 'block';
  loadBaselines();
}
function hideTracker() {
  el.trackerView.style.display = 'none';
  document.querySelector('.layout').style.display = 'flex';
}

async function loadBaselines() {
  el.trackerBody.innerHTML = '<p class="hist-empty">加载可追踪会话……</p>';
  try {
    const data = await (await fetch('/api/tracker/baselines')).json();
    if (data.status !== 'ok') throw new Error(data.message);
    renderBaselines(data.baselines || []);
  } catch (e) {
    el.trackerBody.innerHTML = '<p class="hist-empty">加载失败：' + escHtml(e.message) + '</p>';
  }
}
function renderBaselines(list) {
  if (!list.length) { el.trackerBody.innerHTML = '<p class="hist-empty">还没有可追踪的会话</p>'; return; }
  const wrap = document.createElement('div');
  wrap.className = 'tracker-baselines';
  list.forEach(b => {
    const row = document.createElement('div');
    row.className = 'hist-row';
    const badge = b.cognitive_distortion ? `<span class="hist-badge">${escHtml(b.cognitive_distortion)}</span>` : '';
    row.innerHTML = `<div class="hist-title">${escHtml(b.title || '(空会话)')}</div>
      <div class="hist-meta"><span>${formatDate(b.last_active)}</span><span>· ${b.turn_count} 轮</span>${badge}</div>`;
    const btn = document.createElement('button');
    btn.className = 'btn btn-primary btn-sm';
    btn.textContent = '开始复诊';
    btn.addEventListener('click', () => startCheckin(b.session_id));
    row.appendChild(btn);
    wrap.appendChild(row);
  });
  el.trackerBody.innerHTML = '';
  el.trackerBody.appendChild(wrap);
}

async function startCheckin(baselineId) {
  try {
    const data = await (await fetch('/api/tracker/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ baseline_id: baselineId }),
    })).json();
    if (data.status !== 'ok') throw new Error(data.message);
    Tracker.baselineId = baselineId;
    Tracker.checkinId = data.checkin_id;
    renderCheckinUI(data.question, data.q_index, data.total);
  } catch (e) { showToast('开始复诊失败：' + e.message); }
}

function renderCheckinUI(question, qIndex, total) {
  el.trackerBody.innerHTML = `
    <div class="checkin-progress">复诊中 · 第 ${qIndex + 1}/${total} 问</div>
    <div class="checkin-msgs" id="checkin-msgs"></div>
    <div class="input-bar-inner" style="margin-top:12px">
      <textarea id="checkin-input" class="user-input" rows="1" placeholder="说说你的近况……"></textarea>
      <button class="btn btn-send" id="checkin-send" aria-label="发送">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M22 2L11 13"/><path d="M22 2L15 22l-4-9-9-4 20-7z"/></svg>
      </button>
    </div>`;
  appendCheckinMsg('ai', question);
  $('checkin-send').addEventListener('click', sendCheckin);
  $('checkin-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendCheckin(); }
  });
  $('checkin-input').focus();
}
function appendCheckinMsg(role, text) {
  const box = $('checkin-msgs');
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap ' + (role === 'user' ? 'msg-wrap-user' : 'msg-wrap-ai');
  const row = document.createElement('div');
  row.className = 'msg ' + (role === 'user' ? 'msg-user' : 'msg-ai');
  row.innerHTML = `<div class="msg-avatar">${role === 'user' ? '&#128100;' : '&#9775;'}</div><div class="msg-bubble"></div>`;
  row.querySelector('.msg-bubble').textContent = text;
  wrap.appendChild(row);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}
async function sendCheckin() {
  const input = $('checkin-input');
  const text = input.value.trim();
  if (!text) return;
  appendCheckinMsg('user', text);
  input.value = '';
  input.disabled = true;
  try {
    const data = await (await fetch('/api/tracker/message', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkin_id: Tracker.checkinId, message: text }),
    })).json();
    if (data.status !== 'ok') { showToast(data.message || '出错'); input.disabled = false; return; }
    if (data.done) { renderReport(data.report); }
    else {
      input.disabled = false;
      appendCheckinMsg('ai', data.question);
      const p = document.querySelector('.checkin-progress');
      if (p) p.textContent = `复诊中 · 第 ${data.q_index + 1}/${data.total} 问`;
      input.focus();
    }
  } catch (e) { showToast('发送失败：' + e.message); input.disabled = false; }
}

function renderReport(report) {
  const statusColor = { '改善': 'var(--dot-thought)', '持平': 'var(--text-muted)', '恶化': 'var(--accent)' };
  const rows = (report.form_diff || []).map(d => {
    const label = { same: '不变', changed: '改变', not_reassessed: '未重评', new: '新增' }[d.change] || d.change;
    return `<tr><td>${escHtml(d.dimension)}</td><td>${escHtml(d.before || '—')}</td>
            <td>${escHtml(d.after || '—')}</td><td>${label}</td></tr>`;
  }).join('');
  el.trackerBody.innerHTML = `
    <div class="report-status">整体状态：<b style="color:${statusColor[report.status] || 'inherit'}">${escHtml(report.status)}</b>
      <span class="report-sub">确信度 ${report.baseline_conviction} → ${report.current_conviction}（Δ ${report.conviction_delta >= 0 ? '+' : ''}${report.conviction_delta}，下降为好转）</span></div>
    <div class="report-chart">${renderTrendChart(report.trend || [])}</div>
    <h3 class="report-h">逐维度对比</h3>
    <table class="report-table"><thead><tr><th>维度</th><th>上次</th><th>这次</th><th>变化</th></tr></thead><tbody>${rows}</tbody></table>
    <h3 class="report-h">分析</h3>
    <p class="report-narr">${escHtml(report.narrative || '—')}</p>
    <h3 class="report-h">建议</h3>
    <p class="report-narr">${escHtml(report.suggestion || '—')}</p>
    <button class="btn btn-ghost btn-sm" id="btn-tracker-home" style="margin-top:16px">返回追踪首页</button>`;
  $('btn-tracker-home').addEventListener('click', loadBaselines);
}

/* 无依赖 SVG 折线趋势图。conviction 越低越好，y 轴反向。 */
function renderTrendChart(points) {
  if (!points.length) return '<p class="hist-empty">暂无趋势数据</p>';
  const W = 520, H = 180, pad = 36;
  const xs = points.length > 1 ? points.length - 1 : 1;
  const coords = points.map((p, i) => {
    const x = pad + (W - 2 * pad) * (i / xs);
    const y = pad + (H - 2 * pad) * ((p.conviction || 0) / 100);
    return { x, y, p };
  });
  const line = coords.map((c, i) => (i ? 'L' : 'M') + c.x.toFixed(1) + ' ' + c.y.toFixed(1)).join(' ');
  const dots = coords.map(c =>
    `<circle cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="4" fill="var(--accent)"/>
     <text x="${c.x.toFixed(1)}" y="${(c.y - 10).toFixed(1)}" text-anchor="middle" font-size="11" fill="var(--text-secondary)">${c.p.conviction}</text>
     <text x="${c.x.toFixed(1)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${escHtml(c.p.label)}</text>`
  ).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="确信度趋势图">
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="var(--border)"/>
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="var(--border)"/>
    <path d="${line}" fill="none" stroke="var(--accent-light)" stroke-width="2"/>${dots}</svg>`;
}

/* ── 事件绑定 ────────────────────────────────────────────────── */
el.btnStart.addEventListener('click', async () => {
  const opening = el.welcomeInput.value.trim();
  if (!opening) { el.welcomeInput.focus(); return; }
  const ok = await startSession(opening);
  if (!ok) return;
  showChatUI();
  // 将开场白作为第一条用户消息显示，并直接发送给治疗师
  sendMessage(opening);
});

el.welcomeInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el.btnStart.click(); }
});

el.btnSend.addEventListener('click', () => {
  const msg = el.userInput.value.trim();
  if (msg) sendMessage(msg);
});

el.userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const msg = el.userInput.value.trim();
    if (msg) sendMessage(msg);
  }
});

el.userInput.addEventListener('input', () => autoResize(el.userInput));

el.btnNewSession.addEventListener('click', newSession);
el.btnTogglePanel.addEventListener('click', togglePanel);
el.btnHistory.addEventListener('click', openHistory);
el.btnCloseHistory.addEventListener('click', closeHistory);
el.drawerBackdrop.addEventListener('click', closeHistory);
el.btnExportJson.addEventListener('click', () => exportSession(State.sessionId, 'json'));
el.btnExportMd.addEventListener('click', () => exportSession(State.sessionId, 'md'));
el.btnUndo.addEventListener('click', undoLastTurn);
el.btnTracker.addEventListener('click', showTracker);
el.btnTrackerBack.addEventListener('click', hideTracker);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && el.drawer.classList.contains('show')) closeHistory();
});

/* ── 初始化：隐藏对话区，显示欢迎屏 ─────────────────────────── */
(function init() {
  el.messages.style.display = 'none';
  el.inputBar.style.display = 'none';
  el.welcomeInput.focus();
})();