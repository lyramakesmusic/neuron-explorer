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
  "tabs.showPanel": "show panel",
  "tabs.openAgent": "open conversation",
  "tabs.agentHandle": "‹ {agent}",

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
  "agent.inputTitle": "while Claude works: Enter interrupts and sends, ⌘/Ctrl+Enter queues",
  "agent.send": "send",
  "agent.stop": "stop",
  "agent.explain": "explain {ref}",
  "agent.web": "what does its web do?",
  "agent.webShown": "what does the web of {ref} do?",
  "agent.neurons_one": "1 neuron",
  "agent.neurons_other": "{n} neurons",
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

  // agent tool labels (the expanded call still shows the raw tool name and JSON input, which stay identifiers)
  "tool.overview": "overview",
  "tool.neuron": "neuron",
  "tool.rollouts": "rollouts",
  "tool.cone": "cone",
  "tool.web_summary": "web summary",
  "tool.search_lens": "search lens",
  "tool.top_units": "top units",
  "tool.set_name": "set name",
  "tool.names": "names",
  "tool.steer": "steer",
  "tool.activations": "activations",
  "tool.compare": "compare",
  "tool.logprobs": "logprobs",
  "tool.top_firing": "top firing",
  "tool.camera": "camera",
  "tool.annotate_neurons": "annotate neurons",
  "tool.show_text": "show text",
  "tool.show_search": "show search",
  "tool.open_tab": "open tab",
  "tool.select_neuron": "select neuron",
  "tool.head": "head",
  "tool.neuron_heads": "neuron heads",
  "tool.attention": "attention",
  "tool.attn_drive": "attention drive",
  "tool.select_head": "select head",
  "tool.neuronary_search": "neuronary search",
  "tool.neuronary_get": "neuronary get",
  "tool.neuronary_publish": "neuronary publish",
  "tool.neuronary_update": "neuronary update",
  "tool.neuronary_comment": "neuronary comment",
  "tool.neuronary_reproduce": "neuronary reproduce",
  "tool.neuronary_vote": "neuronary vote",
  "tool.exemplars": "exemplars",
  "tool.neuronary_neuron": "neuronary neuron",
  "tool.neuronary_readouts": "neuronary readouts",
  "tool.neuronary_interpret": "neuronary interpretation",
  "tool.highlight_neurons": "highlight neurons",
  "tool.Read": "Read",
  "tool.Glob": "Glob",
  "tool.Grep": "Grep",
  "tool.Write": "Write",
  "tool.Edit": "Edit",
  "tool.Bash": "Bash",

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
  "lens.jpp": "J++ lens",
  "lens.r": "R lens",
  "lens.s": "S lens · reads",
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
  "results.pass_one": "1 neuron passes · score = activation ÷ its typical extreme",
  "results.pass_other": "{total} neurons pass · score = activation ÷ its typical extreme",
  "results.summary_one": "1 neuron · {tokens} token(s)",
  "results.summary_other": "{total} neurons · {tokens} token(s)",

  "scatter.baseline": "baseline",
  "scatter.random": "random",

  "status.unreachable": "server unreachable",
  "status.idle": "idle",
  "status.loading": "loading",
  "status.indexing": "indexing lens {i}/{n}",
  "status.measuring": "measuring neuron ranges {seen}k/{tokens}k tokens",
  "status.noModel": "no model",
  "status.other": "{raw}",

  "error.http": "{path} · HTTP {status}: {detail}",
  "error.label": "{label}: {error}",
  "error.unknown": "unknown error",
  "error.unreachable": "server unreachable",
  "error.forbidden": "the server refused the request",
  "error.fileTooLarge": "file is too large (over 25 MB)",
  "error.noModel": "the live model is off",
  "error.modelLoading": "the model is loading",
  "error.noSuchChat": "conversation not found",
  "error.noSuchHead": "attention head not found",
  "error.agentExited": "the agent exited ({code})",
  "error.agentExitedNoCode": "the agent exited",
  "error.missingCli": "the agent command-line tool is missing: {name}",
  "error.safety": "the reply was stopped by a safety classifier",
  "error.technical": "technical error: {raw}",
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
  "tabs.showPanel": "显示面板",
  "tabs.openAgent": "打开对话",
  "tabs.agentHandle": "‹ {agent} 对话",

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
  "agent.inputTitle": "Claude 工作时：Enter 中断并发送，⌘/Ctrl+Enter 排队",
  "agent.send": "发送",
  "agent.stop": "停止",
  "agent.explain": "解释 {ref}",
  "agent.web": "它的连接网有什么作用？",
  "agent.webShown": "{ref} 的连接网有什么作用？",
  "agent.neurons_one": "{n} 个神经元",
  "agent.neurons_other": "{n} 个神经元",
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

  "tool.overview": "概览",
  "tool.neuron": "神经元",
  "tool.rollouts": "生成样本",
  "tool.cone": "连接锥",
  "tool.web_summary": "连接网摘要",
  "tool.search_lens": "搜索透镜",
  "tool.top_units": "高排名单元",
  "tool.set_name": "设置名称",
  "tool.names": "已命名神经元",
  "tool.steer": "引导生成",
  "tool.activations": "激活值",
  "tool.compare": "对比读取",
  "tool.logprobs": "对数概率",
  "tool.top_firing": "最强激活",
  "tool.camera": "移动视角",
  "tool.annotate_neurons": "标注神经元",
  "tool.show_text": "显示文本",
  "tool.show_search": "显示搜索",
  "tool.open_tab": "切换标签页",
  "tool.select_neuron": "打开神经元",
  "tool.head": "注意力头",
  "tool.neuron_heads": "神经元的注意力头",
  "tool.attention": "注意力",
  "tool.attn_drive": "注意力驱动",
  "tool.select_head": "打开注意力头",
  "tool.neuronary_search": "搜索 neuronary",
  "tool.neuronary_get": "获取 neuronary 配方",
  "tool.neuronary_publish": "发布 neuronary 配方",
  "tool.neuronary_update": "更新 neuronary 配方",
  "tool.neuronary_comment": "评论 neuronary 配方",
  "tool.neuronary_reproduce": "报告 neuronary 复现",
  "tool.neuronary_vote": "为 neuronary 配方投票",
  "tool.exemplars": "激活示例",
  "tool.neuronary_neuron": "查看 neuronary 神经元",
  "tool.neuronary_readouts": "发送 neuronary 读数",
  "tool.neuronary_interpret": "发布 neuronary 解读",
  "tool.highlight_neurons": "高亮神经元",
  "tool.Read": "读取文件",
  "tool.Glob": "查找文件",
  "tool.Grep": "搜索文件",
  "tool.Write": "写入文件",
  "tool.Edit": "编辑文件",
  "tool.Bash": "命令行",

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
  "lens.jpp": "J++ 透镜",
  "lens.r": "R 透镜",
  "lens.s": "S 透镜 · 读取",
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
  "results.pass_one": "1 个神经元通过 · 得分 = 激活值 ÷ 其典型极值",
  "results.pass_other": "{total} 个神经元通过 · 得分 = 激活值 ÷ 其典型极值",
  "results.summary_one": "1 个神经元 · {tokens} 个 token",
  "results.summary_other": "{total} 个神经元 · {tokens} 个 token",

  "scatter.baseline": "基线",
  "scatter.random": "随机",

  "status.unreachable": "无法连接服务器",
  "status.idle": "空闲",
  "status.loading": "正在加载模型",
  "status.indexing": "正在索引透镜 {i}/{n}",
  "status.measuring": "正在测量神经元范围 {seen}k/{tokens}k token",
  "status.noModel": "未加载模型",
  "status.other": "状态：{raw}",

  "error.http": "{path} · HTTP {status}：{detail}",
  "error.label": "{label}：{error}",
  "error.unknown": "未知错误",
  "error.unreachable": "无法连接服务器",
  "error.forbidden": "服务器拒绝了请求",
  "error.fileTooLarge": "文件过大（超过 25 MB）",
  "error.noModel": "实时模型未启用",
  "error.modelLoading": "模型正在加载",
  "error.noSuchChat": "对话不存在",
  "error.noSuchHead": "注意力头不存在",
  "error.agentExited": "智能体已退出（{code}）",
  "error.agentExitedNoCode": "智能体已退出",
  "error.missingCli": "找不到智能体命令行工具：{name}",
  "error.safety": "回复被安全分类器中止",
  "error.technical": "技术错误：{raw}",
};

const ru = {
  "nav.map": "карта",
  "nav.embeddings": "эмбеддинги",

  "header.layer": "слой",
  "header.layoutTitle": "раскладка листа",
  "header.layoutIndex": "индекс",
  "header.layoutWrites": "записи",
  "header.webTitle": "сеть связей открытого нейрона: конусы входящих и исходящих связей; жёлтые связи тянут читателя вверх, пурпурные — вниз",
  "header.webOff": "без сети",
  "header.webOn": "сеть",
  "header.webZ": "сильнейшие связи сети",
  "header.webZTitle": "сеть: сильнейшие связи открытого нейрона в обе стороны",
  "header.jump": "перейти к нейрону",
  "header.lens": "линза",
  "header.fires": "активации",
  "header.searchLens": "поиск по линзе",
  "header.searchText": "текст: golden gate bridge",
  "header.search": "поиск",
  "header.language": "Язык",

  "view.colorNeuron": "нейрон",
  "view.colorPrompt": "промпт",
  "view.colorDose": "доза",
  "view.colorKind": "тип",

  "focus.close": "закрыть нейрон",
  "focus.name": "имя",
  "focus.featureName": "имя признака",
  "focus.idRandom": "L{layer} случайный {n}",
  "focus.rank": "#{rank}/{n}  масса {mass}  контроль {nullPct}%",

  "list.net": "сеть",
  "list.run": "запуск",
  "list.hide": "скрыть список",
  "list.show": "показать список",
  "list.filter": "фильтр",
  "list.filterAria": "фильтровать по слову или имени",
  "list.random": "случайные",
  "list.infText": "вставьте текст: какие нейроны сильнее всего срабатывают на каждом токене",
  "list.fromLayer": "от слоя",
  "list.toLayer": "до слоя",
  "list.minScore": "минимальный балл",
  "list.lensHas": "линза содержит…",
  "list.lensFilter": "фильтр линзы",

  "tabs.examples": "Примеры",
  "tabs.chat": "Чат",
  "tabs.details": "Детали",
  "tabs.agent": "агент для новых диалогов",
  "tabs.past": "прошлые",
  "tabs.pastTitle": "прошлые диалоги",
  "tabs.new": "новый",
  "tabs.newTitle": "начать новый диалог",
  "tabs.collapse": "свернуть",
  "tabs.showPanel": "показать панель",
  "tabs.openAgent": "открыть диалог",
  "tabs.agentHandle": "‹ {agent}",

  "chat.add": "сложить",
  "chat.writes": "пишет",
  "chat.reads": "читает вас",
  "chat.smaller": "меньше",
  "chat.larger": "больше",
  "chat.dose": "доза",
  "chat.clear": "очистить",
  "chat.who": "Кто вы?",
  "chat.textToRead": "текст для чтения",
  "chat.run": "запустить",
  "chat.stop": "стоп",

  "agent.empty": "Спросите что угодно о модели или откройте нейрон и нажмите <b>объяснить</b>.",
  "agent.attach": "прикрепить",
  "agent.attachTitle": "прикрепить файлы (или перетащите / вставьте)",
  "agent.ask": "Спросить Claude…",
  "agent.inputTitle": "пока Claude работает: Enter прерывает и отправляет, ⌘/Ctrl+Enter ставит в очередь",
  "agent.send": "отправить",
  "agent.stop": "стоп",
  "agent.explain": "объяснить {ref}",
  "agent.web": "что делает его сеть связей?",
  "agent.webShown": "что делает сеть связей {ref}?",
  "agent.neurons_one": "{n} нейрон",
  "agent.neurons_few": "{n} нейрона",
  "agent.neurons_many": "{n} нейронов",
  "agent.running": "выполняется…",
  "agent.stopped": "остановлено",
  "agent.edit": "изменить",
  "agent.editTitle": "удалить это сообщение и ответ и вернуть его в поле ввода",
  "agent.reroll": "↻ заново",
  "agent.rerollTitle": "отбросить этот ответ и спросить заново",
  "agent.useName": "использовать имя: {name}",
  "agent.attached": "(вложенные файлы)",
  "agent.pasted": "вставлено",
  "agent.remove": "удалить",
  "agent.minutesAgo": "{n} мин назад",
  "agent.hoursAgo": "{n} ч назад",
  "agent.messages_one": "{n} сообщение",
  "agent.messages_few": "{n} сообщения",
  "agent.messages_many": "{n} сообщений",

  "tool.overview": "обзор",
  "tool.neuron": "нейрон",
  "tool.rollouts": "генерации",
  "tool.cone": "конус",
  "tool.web_summary": "сводка сети",
  "tool.search_lens": "поиск по линзе",
  "tool.top_units": "топ юнитов",
  "tool.set_name": "задать имя",
  "tool.names": "именованные нейроны",
  "tool.steer": "управление генерацией",
  "tool.activations": "активации",
  "tool.compare": "сравнение",
  "tool.logprobs": "логвероятности",
  "tool.top_firing": "сильнейшие активации",
  "tool.camera": "камера",
  "tool.annotate_neurons": "аннотировать нейроны",
  "tool.show_text": "показать текст",
  "tool.show_search": "показать поиск",
  "tool.open_tab": "открыть вкладку",
  "tool.select_neuron": "выбрать нейрон",
  "tool.head": "голова внимания",
  "tool.neuron_heads": "головы внимания нейрона",
  "tool.attention": "внимание",
  "tool.attn_drive": "вклад внимания",
  "tool.select_head": "выбрать голову",
  "tool.neuronary_search": "поиск в neuronary",
  "tool.neuronary_get": "получить рецепт neuronary",
  "tool.neuronary_publish": "опубликовать рецепт neuronary",
  "tool.neuronary_update": "обновить рецепт neuronary",
  "tool.neuronary_comment": "прокомментировать рецепт neuronary",
  "tool.neuronary_reproduce": "сообщить о воспроизведении neuronary",
  "tool.neuronary_vote": "голосовать за рецепт neuronary",
  "tool.exemplars": "примеры активаций",
  "tool.neuronary_neuron": "нейрон neuronary",
  "tool.neuronary_readouts": "отправить показания neuronary",
  "tool.neuronary_interpret": "опубликовать интерпретацию neuronary",
  "tool.highlight_neurons": "выделить нейроны",
  "tool.Read": "чтение файла",
  "tool.Glob": "поиск файлов",
  "tool.Grep": "поиск по файлам",
  "tool.Write": "запись файла",
  "tool.Edit": "правка файла",
  "tool.Bash": "командная строка",

  "map.allLayers": "‹ все слои  ·  L{l}",
  "map.fullAttention": "полное внимание · {q} q / {kv} kv голов",
  "map.linearAttention": "линейное внимание · {k} k / {v} v голов",
  "map.mass": "масса {z}",

  "web.title": "сеть: сильнейшие связи открытого нейрона в обе стороны (дальше — по {k} на шаг) · связей: {n}",
  "web.head": "сеть L{layer} N{n} · связей: {links} · {src}, по превышению над уровнем слоя",
  "web.srcSteer": "линза управления",
  "web.srcLogit": "логит-линза",
  "web.fedBy": "питается от · входящих {n}",
  "web.heldBy": "сдерживается",
  "web.toward": "питает → толкает к · исходящих {n}",
  "web.away": "питает → толкает от",
  "web.nothing": "ничего выше случайного уровня",
  "web.chipTitle": "нейронов: {n} · балл {w}",

  "lens.heads": "головы",
  "lens.steer": "линза управления · KL {up} / {down}",
  "lens.logit": "логит-линза",
  "lens.jpp": "J++-линза",
  "lens.r": "R-линза",
  "lens.s": "S-линза · читает",
  "lens.steerShort": "линза управления",
  "lens.push": "+ толчок · KL {kl}",
  "lens.pull": "− толчок · KL {kl}",

  "details.words": "слова",
  "details.dose": "доза",
  "details.push": "толчок",
  "details.distinct": "distinct-4",
  "details.capped": "ограничено",
  "details.logprob": "logprob",
  "details.connections": "связи",
  "details.writesInto": "записывает в",
  "details.drivenBy": "управляется",
  "details.headsReading": "головы, читающие его",
  "details.headsFeeding": "головы, подающие в него",
  "details.rollouts": "прогонов: {n} · z {z} · {prompts}",

  "results.indexing": "индексация линзы",
  "results.running": "выполняется…",
  "results.measuring": "измерение диапазонов нейронов",
  "results.pass_one": "1 нейрон проходит · балл = активация ÷ его типичный экстремум",
  "results.pass_few": "{total} нейрона проходят · балл = активация ÷ его типичный экстремум",
  "results.pass_many": "{total} нейронов проходят · балл = активация ÷ его типичный экстремум",
  "results.summary_one": "1 нейрон · {tokens} ток.",
  "results.summary_few": "{total} нейрона · {tokens} ток.",
  "results.summary_many": "{total} нейронов · {tokens} ток.",

  "scatter.baseline": "базовая",
  "scatter.random": "случайная",

  "status.unreachable": "сервер недоступен",
  "status.idle": "простой",
  "status.loading": "загрузка",
  "status.indexing": "индексация линзы {i}/{n}",
  "status.measuring": "измерение диапазонов нейронов {seen}k/{tokens}k токенов",
  "status.noModel": "нет модели",
  "status.other": "статус: {raw}",

  "error.http": "{path} · HTTP {status}: {detail}",
  "error.label": "{label}: {error}",
  "error.unknown": "неизвестная ошибка",
  "error.unreachable": "сервер недоступен",
  "error.forbidden": "сервер отклонил запрос",
  "error.fileTooLarge": "файл слишком большой (больше 25 МБ)",
  "error.noModel": "живая модель выключена",
  "error.modelLoading": "модель загружается",
  "error.noSuchChat": "диалог не найден",
  "error.noSuchHead": "голова внимания не найдена",
  "error.agentExited": "агент завершился ({code})",
  "error.agentExitedNoCode": "агент завершился",
  "error.missingCli": "не найдена командная утилита агента: {name}",
  "error.safety": "ответ остановлен классификатором безопасности",
  "error.technical": "техническая ошибка: {raw}",
};

const DICT = { en, "zh-CN": zh, ru };
const STORE_KEY = "ne.lang";

let current = "en";
let selectEl = null;

// Russian needs one / few / many; the other dictionaries use one / other.
const ruPlural = typeof Intl !== "undefined" && Intl.PluralRules ? new Intl.PluralRules("ru") : null;

const normalize = lang => (lang === "zh-CN" ? "zh-CN" : /^ru(-|$)/i.test(lang) ? "ru" : "en");

function storedLang() {
  try { return localStorage.getItem(STORE_KEY); } catch { return null; }
}

function detectLang() {
  // only claim Simplified Chinese: a Traditional Chinese browser gets the English UI
  const primary = navigator.languages?.[0] || navigator.language || "";
  if (/^zh(-(cn|sg|hans)(-|$)|$)/i.test(primary)) return "zh-CN";
  return /^ru(-|$)/i.test(primary) ? "ru" : "en";
}

export function getLang() { return current; }

export function t(key, vars) {
  const text = DICT[current]?.[key] ?? en[key] ?? key;
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (m, k) => (vars[k] ?? m));
}

// English needs a plural for a few counted labels; Chinese uses the same form for both, Russian three forms.
export function tp(key, n, vars = {}) {
  if (current === "ru" && ruPlural) {
    // fractions select "other", which Russian lacks here, so they read as "many"
    const form = ruPlural.select(n);
    const suffix = ["one", "few", "many"].includes(form) ? form : "many";
    const name = [`${key}_${suffix}`, `${key}_many`, `${key}_one`, `${key}_other`].find(k => DICT.ru[k] != null) || `${key}_one`;
    return t(name, { ...vars, n });
  }
  return t(`${key}_${n === 1 ? "one" : "other"}`, { ...vars, n });
}

// Backend and network errors are stable English strings (or raw technical output). Map the common ones here so the
// API stays language-neutral; anything unmatched stays raw in English and gets a prefix in the other languages.
function knownError(raw) {
  const text = raw.trim();
  if (!text) return t("error.unknown");
  if (text === "server unreachable" || /^(failed to fetch|load failed|fetch failed|networkerror|network error)/i.test(text)
    || /internet connection appears to be offline|network connection was lost/i.test(text)) return t("error.unreachable");
  if (/^forbidden\b/i.test(text)) return t("error.forbidden");
  if (/file over 25 mb|file too large|payload too large|request entity too large/i.test(text)) return t("error.fileTooLarge");
  if (/stopped by (?:the )?[\w -]*safety classifier/i.test(text)) return t("error.safety");
  if (/no such chat|conversation not found|chat not found|no such conversation/i.test(text)) return t("error.noSuchChat");
  if (/no such head|head not found/i.test(text)) return t("error.noSuchHead");
  const missing = text.match(/No such file or directory[^'"]*['"]([^'"]+)['"]|executable file not found[^'"]*['"]?([^'"\s]+)['"]?|spawn ([^ ]+) ENOENT/i);
  if (missing) return t("error.missingCli", { name: (missing[1] || missing[2] || missing[3] || "").split("/").pop() });
  const exited = text.match(/\b(?:claude|codex|dsh) exited(?:\s+(\d+))?\b/i);
  if (exited) return exited[1] ? t("error.agentExited", { code: exited[1] }) : t("error.agentExitedNoCode");
  if (/model loading|loading model/i.test(text)) return t("error.modelLoading");
  if (/no live model|live model is off|live model disabled|explorer_no_model|^no model$/i.test(text)) return t("error.noModel");
  return null;
}

export function localizeError(input) {
  const raw = String(input?.message ?? input ?? "").trim();
  const known = knownError(raw);
  if (known) return known;
  const http = raw.match(/^(\S+) (\d{3}):\s*([\s\S]*)$/);
  if (http) {
    const [, path, status, body] = http;
    let detail = body;
    try { const j = JSON.parse(body); if (j && typeof j.error === "string") detail = j.error; } catch {}
    detail = knownError(detail) || (getLang() !== "en" ? t("error.technical", { raw: detail }) : detail);
    return t("error.http", { path, status, detail });
  }
  return getLang() !== "en" ? t("error.technical", { raw }) : raw;
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
