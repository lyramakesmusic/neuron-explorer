// Tiny i18n layer for the explorer UI: dictionary lookup, DOM translation and a persisted language choice.
// English is the source of truth. A new language needs a dictionary here, an <option> in index.html, and a branch
// in normalize() and detectLang().

const en = {
  "nav.map": "map",
  "nav.embeddings": "embeddings",

  "header.layer": "layer",
  "header.layoutTitle": "sheet layout",
  "header.layoutIndex": "index",
  "header.layoutWrites": "writes",
  "header.webTitle": "connection web of the focused neuron: upstream and downstream cones; yellow links push the reader up, magenta push it down",
  "header.webOff": "no web",
  "header.webOn": "web",
  "header.webZ": "web top links",
  "header.webZTitle": "web: the open neuron's top links each way",
  "header.jump": "go to neuron",
  "header.lens": "lens",
  "header.fires": "fires",
  "header.searchLens": "search lens",
  "header.searchText": "text: golden gate bridge",
  "header.search": "search",
  "header.language": "Language",

  "view.colorNeuron": "neuron",
  "view.colorPrompt": "prompt",
  "view.colorDose": "dose",
  "view.colorKind": "kind",

  "focus.close": "close neuron",
  "focus.name": "name",
  "focus.featureName": "feature name",
  "focus.idRandom": "L{layer} random {n}",
  "focus.rank": "#{rank}/{n}  mass {mass}  null {nullPct}%",

  "list.net": "net",
  "list.run": "run",
  "list.hide": "hide list",
  "list.show": "show list",
  "list.filter": "filter",
  "list.filterAria": "filter by word or name",
  "list.random": "random",
  "list.infText": "paste text: see which neurons fire most on each token",
  "list.fromLayer": "from layer",
  "list.toLayer": "to layer",
  "list.minScore": "minimum score",
  "list.lensHas": "lens has…",
  "list.lensFilter": "lens filter",

  "tabs.examples": "Examples",
  "tabs.chat": "Chat",
  "tabs.details": "Details",
  "tabs.agent": "agent for new conversations",
  "tabs.past": "past",
  "tabs.pastTitle": "past conversations",
  "tabs.new": "new",
  "tabs.newTitle": "start a new conversation",
  "tabs.collapse": "collapse",
  "tabs.showNeuron": "show neuron",

  "chat.add": "add",
  "chat.writes": "writes",
  "chat.reads": "reads you",
  "chat.smaller": "smaller",
  "chat.larger": "larger",
  "chat.dose": "dose",
  "chat.clear": "clear",
  "chat.who": "Who are you?",
  "chat.textToRead": "text to read",
  "chat.run": "run",
  "chat.stop": "stop",

  "agent.empty": "Ask anything about the model, or open a neuron and press <b>explain</b>.",
  "agent.attach": "attach",
  "agent.attachTitle": "attach files (or drop / paste them)",
  "agent.ask": "Ask Claude…",
  "agent.ask.claude": "Ask Claude…",
  "agent.ask.codex": "Ask Codex…",
  "agent.ask.dsh": "Ask DSH…",
  "agent.inputTitle.claude": "while Claude works: Enter interrupts and sends, ⌘/Ctrl+Enter queues",
  "agent.inputTitle.codex": "while Codex works: Enter sends the message into the running turn",
  "agent.inputTitle.dsh": "while DSH works: Enter queues the message for after the current reply",
  "agent.err.dsh_missing": "DSH is not installed or not on PATH. Install DeepSeek Harness first.",
  "agent.err.dsh_no_api_key": "DSH needs a DeepSeek API key: set DEEPSEEK_API_KEY, or save it as refs.DEEPSEEK_API_KEY in ~/.dsh/.credentials.yaml.",
  "agent.err.dsh_profile": "DSH profile setup failed. Check that `dsh --version` works and that its profile can install packages.",
  "agent.inputTitle": "while Claude works: Enter interrupts and sends, ⌘/Ctrl+Enter queues",
  "agent.send": "send",
  "agent.stop": "stop",
  "agent.explain": "explain {ref}",
  "agent.web": "what does its web do?",
  "agent.webShown": "what does the web of {ref} do?",
  "agent.neurons": "{n} neurons",
  "agent.running": "running…",
  "agent.stopped": "stopped",
  "agent.edit": "edit",
  "agent.editTitle": "remove this message and its reply, and put it back in the box",
  "agent.reroll": "↻ reroll",
  "agent.rerollTitle": "throw this reply away and ask again",
  "agent.useName": "use name: {name}",
  "agent.attached": "(attached files)",
  "agent.pasted": "pasted",
  "agent.remove": "remove",
  "agent.minutesAgo": "{n}m ago",
  "agent.hoursAgo": "{n}h ago",
  "agent.messages_one": "{n} message",
  "agent.messages_other": "{n} messages",

  "map.allLayers": "‹ all layers  ·  L{l}",
  "map.fullAttention": "full attention · {q} q / {kv} kv heads",
  "map.linearAttention": "linear attention · {k} k / {v} v heads",
  "map.mass": "mass {z}",

  "web.title": "web: the open neuron's top links each way (later hops follow {k}) · {n} links",
  "web.head": "web of L{layer} N{n} · {links} links · {src}, ranked by lift over each layer",
  "web.srcSteer": "steer lens",
  "web.srcLogit": "logit lens",
  "web.fedBy": "fed by · {n} upstream neurons",
  "web.heldBy": "held back by",
  "web.toward": "feeds → pushes toward · {n} downstream neurons",
  "web.away": "feeds → pushes away from",
  "web.nothing": "nothing above chance",
  "web.chipTitle": "{n} neurons · score {w}",

  "lens.heads": "heads",
  "lens.steer": "steer lens · KL {up} / {down}",
  "lens.logit": "logit lens",
  "lens.steerShort": "steer lens",
  "lens.push": "+ push · KL {kl}",
  "lens.pull": "− push · KL {kl}",

  "details.words": "words",
  "details.dose": "dose",
  "details.push": "push",
  "details.distinct": "distinct-4",
  "details.capped": "capped",
  "details.logprob": "logprob",
  "details.connections": "connections",
  "details.writesInto": "writes into",
  "details.drivenBy": "driven by",
  "details.headsReading": "heads reading it",
  "details.headsFeeding": "heads feeding it",
  "details.rollouts": "{n} rollouts · z {z} · {prompts}",

  "results.indexing": "indexing lens",
  "results.running": "running…",
  "results.measuring": "measuring neuron ranges",
  "results.pass": "{total} neurons pass · score = activation ÷ its typical extreme",
  "results.summary": "{total} neurons · {tokens} tokens",

  "scatter.baseline": "baseline",
  "scatter.random": "random",

  "status.unreachable": "server unreachable",
  "status.idle": "idle",
  "status.loading": "loading",
  "status.indexing": "indexing lens {i}/{n}",
  "status.measuring": "measuring neuron ranges {seen}k/{tokens}k tokens",
  "status.noModel": "no model",
};

const zh = {
  "nav.map": "地图",
  "nav.embeddings": "嵌入图",

  "header.layer": "层",
  "header.layoutTitle": "神经元排列",
  "header.layoutIndex": "序号",
  "header.layoutWrites": "写入",
  "header.webTitle": "聚焦神经元的连接网：上游与下游锥体；黄色连接将接收方推高，品红将其压低",
  "header.webOff": "隐藏",
  "header.webOn": "显示",
  "header.webZ": "连接网最强连接数",
  "header.webZTitle": "连接网：打开神经元的双向最强连接",
  "header.jump": "跳转到神经元",
  "header.lens": "透镜",
  "header.fires": "激活",
  "header.searchLens": "搜索透镜",
  "header.searchText": "文本：金门大桥",
  "header.search": "搜索",
  "header.language": "语言",

  "view.colorNeuron": "神经元",
  "view.colorPrompt": "提示词",
  "view.colorDose": "剂量",
  "view.colorKind": "类型",

  "focus.close": "关闭神经元",
  "focus.name": "名称",
  "focus.featureName": "特征名称",
  "focus.idRandom": "L{layer} 随机 {n}",
  "focus.rank": "#{rank}/{n}  簇质量 {mass}  对照分位 {nullPct}%",

  "list.net": "网络",
  "list.run": "运行",
  "list.hide": "隐藏列表",
  "list.show": "显示列表",
  "list.filter": "筛选",
  "list.filterAria": "按词或名称筛选",
  "list.random": "随机",
  "list.infText": "粘贴文本：查看每个 token 上激活最强的神经元",
  "list.fromLayer": "起始层",
  "list.toLayer": "结束层",
  "list.minScore": "最低得分",
  "list.lensHas": "透镜包含…",
  "list.lensFilter": "透镜筛选",

  "tabs.examples": "示例",
  "tabs.chat": "对话",
  "tabs.details": "详情",
  "tabs.agent": "新对话使用的智能体",
  "tabs.past": "历史",
  "tabs.pastTitle": "历史对话",
  "tabs.new": "新建",
  "tabs.newTitle": "开始新对话",
  "tabs.collapse": "收起",
  "tabs.showNeuron": "显示神经元",

  "chat.add": "加",
  "chat.writes": "写入",
  "chat.reads": "读入",
  "chat.smaller": "减小",
  "chat.larger": "增大",
  "chat.dose": "剂量",
  "chat.clear": "清空",
  "chat.who": "你是谁？",
  "chat.textToRead": "要读取的文本",
  "chat.run": "运行",
  "chat.stop": "停止",

  "agent.empty": "可以询问模型的任何问题，或打开一个神经元并点击“<b>解释</b>”。",
  "agent.attach": "附加",
  "agent.attachTitle": "附加文件（也可拖入或粘贴）",
  "agent.ask": "询问 Claude…",
  "agent.ask.claude": "询问 Claude…",
  "agent.ask.codex": "询问 Codex…",
  "agent.ask.dsh": "询问 DSH…",
  "agent.inputTitle.claude": "Claude 工作时：Enter 中断并发送，⌘/Ctrl+Enter 排队",
  "agent.inputTitle.codex": "Codex 工作时：Enter 把消息送入当前回合",
  "agent.inputTitle.dsh": "DSH 工作时：Enter 把消息排到当前回复之后",
  "agent.err.dsh_missing": "未安装 DSH 或不在 PATH 中。请先安装 DeepSeek Harness。",
  "agent.err.dsh_no_api_key": "DSH 需要 DeepSeek API key：请设置 DEEPSEEK_API_KEY，或在 ~/.dsh/.credentials.yaml 的 refs.DEEPSEEK_API_KEY 中保存。",
  "agent.err.dsh_profile": "DSH profile 初始化失败。请检查 `dsh --version` 是否正常，以及该 profile 能否安装依赖。",
  "agent.inputTitle": "Claude 工作时：Enter 中断并发送，⌘/Ctrl+Enter 排队",
  "agent.send": "发送",
  "agent.stop": "停止",
  "agent.explain": "解释 {ref}",
  "agent.web": "它的连接网有什么作用？",
  "agent.webShown": "{ref} 的连接网有什么作用？",
  "agent.neurons": "{n} 个神经元",
  "agent.running": "运行中…",
  "agent.stopped": "已停止",
  "agent.edit": "编辑",
  "agent.editTitle": "删除这条消息及其回复，并放回输入框",
  "agent.reroll": "↻ 重新生成",
  "agent.rerollTitle": "丢弃这条回复并重新提问",
  "agent.useName": "使用名称：{name}",
  "agent.attached": "（附件）",
  "agent.pasted": "粘贴的文件",
  "agent.remove": "移除",
  "agent.minutesAgo": "{n} 分钟前",
  "agent.hoursAgo": "{n} 小时前",
  "agent.messages_one": "{n} 条消息",
  "agent.messages_other": "{n} 条消息",

  "map.allLayers": "‹ 全部层 · L{l}",
  "map.fullAttention": "全注意力 · {q} 个 q / {kv} 个 kv 头",
  "map.linearAttention": "线性注意力 · {k} 个 k / {v} 个 v 头",
  "map.mass": "簇质量 {z}",

  "web.title": "连接网：当前神经元的双向最强连接（后续跳数每跳取前 {k} 条）· {n} 条连接",
  "web.head": "L{layer} N{n} 的连接网 · {links} 条连接 · {src}，按各层提升度排序",
  "web.srcSteer": "引导透镜",
  "web.srcLogit": "logit 透镜",
  "web.fedBy": "由 {n} 个上游神经元输入",
  "web.heldBy": "抑制来自",
  "web.toward": "输出 · 推向 {n} 个下游神经元",
  "web.away": "输出 · 推离",
  "web.nothing": "未高于随机水平",
  "web.chipTitle": "{n} 个神经元 · 得分 {w}",

  "lens.heads": "注意力头",
  "lens.steer": "引导透镜 · KL {up} / {down}",
  "lens.logit": "logit 透镜",
  "lens.steerShort": "引导透镜",
  "lens.push": "+ 推动 · KL {kl}",
  "lens.pull": "− 推动 · KL {kl}",

  "details.words": "词",
  "details.dose": "剂量",
  "details.push": "推动",
  "details.distinct": "distinct-4",
  "details.capped": "截断率",
  "details.logprob": "logprob",
  "details.connections": "连接",
  "details.writesInto": "写入到",
  "details.drivenBy": "驱动自",
  "details.headsReading": "读取它的注意力头",
  "details.headsFeeding": "为其输入的注意力头",
  "details.rollouts": "{n} 次生成 · z {z} · {prompts}",

  "results.indexing": "正在索引透镜",
  "results.running": "运行中…",
  "results.measuring": "正在测量神经元范围",
  "results.pass": "{total} 个神经元通过 · 得分 = 激活值 ÷ 其典型极值",
  "results.summary": "{total} 个神经元 · {tokens} 个 token",

  "scatter.baseline": "基线",
  "scatter.random": "随机",

  "status.unreachable": "无法连接服务器",
  "status.idle": "空闲",
  "status.loading": "正在加载模型",
  "status.indexing": "正在索引透镜 {i}/{n}",
  "status.measuring": "正在测量神经元范围 {seen}k/{tokens}k token",
  "status.noModel": "未加载模型",
};

const DICT = { en, "zh-CN": zh };
const STORE_KEY = "ne.lang";

let current = "en";
let selectEl = null;

const normalize = lang => (lang === "zh-CN" ? "zh-CN" : "en");

function storedLang() {
  try { return localStorage.getItem(STORE_KEY); } catch { return null; }
}

function detectLang() {
  // only claim Simplified Chinese: a Traditional Chinese browser gets the English UI
  const primary = navigator.languages?.[0] || navigator.language || "";
  return /^zh(-(cn|sg|hans)(-|$)|$)/i.test(primary) ? "zh-CN" : "en";
}

export function getLang() { return current; }

export function t(key, vars) {
  const text = DICT[current]?.[key] ?? en[key] ?? key;
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (m, k) => (vars[k] ?? m));
}

// English needs a plural for a few counted labels; Chinese uses the same form for both.
export function tp(key, n, vars = {}) {
  return t(`${key}_${n === 1 ? "one" : "other"}`, { ...vars, n });
}

// Translate the static markup: data-i18n (text), data-i18n-html (markup), and the placeholder/title/aria variants.
export function applyTranslations(root = document) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
  for (const el of root.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of root.querySelectorAll("[data-i18n-aria-label]")) el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
}

function apply() {
  document.documentElement.lang = current;
  applyTranslations();
}

export function setLang(lang) {
  const next = normalize(lang);
  if (next === current) return;
  current = next;
  try { localStorage.setItem(STORE_KEY, current); } catch {}
  if (selectEl) selectEl.value = current;
  apply();
  document.dispatchEvent(new CustomEvent("i18nchange", { detail: { lang: current } }));
}

// Call once, after the DOM is parsed, with the language <select> (optional).
export function initI18n(select) {
  selectEl = select || null;
  current = normalize(storedLang() || detectLang());
  if (selectEl) {
    selectEl.value = current;
    selectEl.addEventListener("change", () => setLang(selectEl.value));
  }
  apply();
}
