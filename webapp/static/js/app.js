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

/* ── 初始化：隐藏对话区，显示欢迎屏 ─────────────────────────── */
(function init() {
  el.messages.style.display = 'none';
  el.inputBar.style.display = 'none';
  el.welcomeInput.focus();
})();