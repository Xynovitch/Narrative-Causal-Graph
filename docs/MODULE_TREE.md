# CEKG 模块树状结构

> Causal Event Knowledge Graph — 从叙事文本中提取结构化因果知识图谱的 Python 流水线。
> 全项目共 6 个大模块 → 21 个中模块 → 53 个小模块。

---

## Level 1: 大模块（6 个）

```
CEKG 流水线
├── [A] 入口与编排层
├── [B] 本体与配置层
├── [C] 数据模型与基础设施层
├── [D] 文本处理与事件抽取层
├── [E] 图谱构建与因果推理层
└── [F] 主题标注与导出层
```

---

## Level 2 → Level 3: 中模块 → 小模块

### [A] 入口与编排层 ─── 整个流水线的控制中枢

```
[A] 入口与编排层
│
├── [A1] CLI 入口 · main.py
│   │   负责解析命令行参数、初始化流水线编排器并启动异步执行。是用户与 CEKG 之间
│   │   的唯一交互界面，通过 argparse 提供 `--input`、`--full`/`--fast` 预设模式、
│   │   `--resume` 断点续传、`--max-pairs` 等全量可调参数。
│   │
│   └── [A1.1] main()
│           入口函数，解析命令行参数（输入路径、输出目录、checkpoint 管理、
│           特征开关如 `--disable-mixed-theory`、LLM 后端选择等），实例化
│           CEKGPreprocessor，调用 `run_async()` 执行完整流水线并输出结果摘要。
│
├── [A2] 流水线主编排器 · pipeline.py（959 行，整个项目最核心的文件）
│   │   定义 CEKGPreprocessor 类和所有 8 个处理阶段的编排逻辑。协调 LLM 服务调用、
│   │   checkpoint 系统、上下文传播、因果链接和主题标注之间的数据流转。每阶段
│   │   支持独立的断点保存和恢复，阶段间数据通过 Python 对象直接传递。是项目
│   │   的骨架——所有其他模块都以它为枢纽参与协作。
│   │
│   ├── [A2.1] CEKGPreprocessor
│   │          主编排类，管理 8 个处理阶段的生命周期。在 `__init__` 中接收所有
│   │          配置参数（输入路径、模式、候选对数上限、主题阈值、并发数等），
│   │          通过 `run_async()` 串联 Stage0–7 的完整执行流程。每个阶段完成后
│   │          调用 CheckpointManager 保存状态快照，支持从中断处恢复而不重复消费
│   │          API 额度。
│   │
│   ├── [A2.2] run_async()
│   │          异步执行入口，依次调用各 Stage 方法。从文本加载（Stage1）开始，
│   │          经事件抽取（Stage2）、上下文传播（Stage3）、Agent 分类（Stage4）、
│   │          场景分组（Stage5）、因果链接（Stage6）、主题标注（Stage7），
│   │          最后调用 exporters 输出 JSON-LD / Cypher / CSV。
│   │
│   ├── [A2.3] _process_chapter_chunked()
│   │          按章节分块抽取事件的核心方法。将长章节用 wtpsplit 按 SAT 句子
│   │          分割，组装为带重叠窗口的 chunk，通过 asyncio.Semaphore 控制并发
│   │          调用 LLM 批量抽取。含独立的重试机制和进度日志。
│   │
│   ├── [A2.4] _parse_event_json_data()
│   │          解析 LLM 返回的事件 JSON，创建 CEKEvent 数据对象。进行字段补全、
│   │          默认值填充、理论归属标记，并调用 CoreferenceResolver 规整角色名。
│   │
│   └── [A2.5] normalize_theory_name()
│              将混合大小写的理论名称（如 "McKee"、"truby"）统一为带 @ 前缀的
│              规范形式（@McKee / @Truby），用于 schema 查询时的理论过滤。
│
└── [A3] 项目文档
    │   入口层的文档体系，帮助开发者和用户理解项目架构与使用方法。
    │
    ├── [A3.1] README.md
    │         项目总览文档，涵盖流水线架构图、输出产物表、成本估算、Setup 指南、
    │         Graph Explorer 使用说明和 flag 参考。
    │
    └── [A3.2] CHANGELOG.md
              跨年度开发变更记录，记录了 SemanticLink→ThematicLink 架构重构、
              --full 模式 max-pairs 修复、Agent 分类 schema 自动检测等关键变更。
```

### [B] 本体与配置层 ─── 叙事理论的机器可读形式化

```
[B] 本体与配置层
│
├── [B1] 核心叙事本体 · schema.json（2,467 行）
│   │   整个流水线的知识底座。定义了 47 种 Agent 类型（PROTAGONIST_HERO、
│   │   MORAL_ANTAGONIST、REVELATION_GIVER 等）、McKee 和 Truby 两套因果
│   │   关系类型（DIRECT_CAUSE、ENABLES、PREVENTS、MORAL_CHALLENGE、
│   │   TRIGGERS、CONCEALS、REDEEMS 等）、5 种结构主题（POWER/WEALTH/
│   │   KINSHIP/JUSTICE/KNOWLEDGE）、事件类型分类体系、地点类型词典和
│   │   时间类型词典。LLM 所有 prompt 注入均以此为本体论约束。
│   │
│   └── （无独立子模块，为声明式 JSON 配置文件）
│
├── [B2] 事件类型本体 · event_ontology.json
│   │   定义 98 种事件标签分类体系（PHYSICAL_ACTION、SPEECH_ACT、MENTAL_STATE、
│   │   SOCIAL_INTERACTION、EMOTIONAL_EXPRESSION 等），为 LLM 事件抽取提供
│   │   受控词汇表。由 scripts/generate_ontology.py 从 13 部小说 LLM 分析结果
│   │   中自底向上提取生成。
│   │
│   └── （无独立子模块，为声明式 JSON 配置文件）
│
├── [B3] 关系类型本体 · relationship_ontology.json
│   │   定义 106 种关系类型分类（CAUSES、FOLLOWS、MOTIVATES、HOSTS 等），
│   │   为因果链接和主题标注提供候选关系词汇。由 scripts/generate_relationship_
│   │   ontology.py 自顶向下通过 LLM 生成，与 event_ontology.json 互补。
│   │
│   └── （无独立子模块，为声明式 JSON 配置文件）
│
├── [B4] 本体加载器 · ontology_loader.py（247 行）
│   │   将 schema.json 及附属本体加载为运行时可用字典。提供 OntologyManager
│   │   单例模式管理器，支持 JSON 拼接修复、自定义键名兼容、理论过滤查询、
│   │   类型验证等。流水线所有阶段共享同一本体管理器实例，确保一致性。
│   │
│   ├── [B4.1] OntologyManager
│   │          本体管理器核心类。从 JSON schema 文件加载事件类型、关系类型、
│   │          Agent 类型、地点类型和时间类型的字典映射。提供按理论过滤的
│   │          查询接口（如只取 McKee 因果关系）、类型存在性校验和分类查询。
│   │
│   ├── [B4.2] get_ontology_manager()
│   │          获取本体管理器单例。首次调用时加载 schema，后续调用返回缓存实例，
│   │          避免重复 I/O 和 JSON 解析开销。
│   │
│   └── [B4.3] EventType / RelationType / AgentType
│              轻量数据类，封装本体条目的结构化信息（名称、描述、所属理论、
│              父类别等），用于类型安全的查询和 Prompt 生成。
│
└── [B5] 环境配置 · config.py（34 行）
    │   通过 python-dotenv 加载 .env 文件中的 LLM 连接参数（VLLM_BASE_URL、
    │   VLLM_MODEL、OPENAI_API_KEY、OPENAI_MODEL 等），并定义全局运行参数：
    │   批处理大小（BATCH_SIZE）、缓存上限（CACHE_MAX_SIZE）、受控行为
    │   本体映射（CONTROLLED_BEHAVIOR_ONTOLOGY）等。
    │
    └── （所有导出为模块级常量，无独立函数/类）
```

### [C] 数据模型与基础设施层 ─── 类型系统与工具支撑

```
[C] 数据模型与基础设施层
│
├── [C1] 核心数据类 · schemas.py（142 行）
│   │   定义 CEKG 的全部数据结构和类型系统。所有流水线阶段的数据交换均以
│   │   这些 dataclass 为契约，确保从 LLM 输出解析到导出之间的类型安全。
│   │
│   ├── [C1.1] CEKEvent
│   │          因果事件节点数据类。核心字段包括 event_id、source_quote（原文引用）、
│   │          actors（主动参与者）、patients（被动承受者）、why_factors（动机因子）、
│   │          place/location、time、chapter、sequence（章节内序号）、
│   │          event_type（McKee 或 Truby 理论分类）、theme_annotations（五主题
│   │          参与度字典）和 theory 归属标记。
│   │
│   ├── [C1.2] CausalLink
│   │          因果边数据类。记录 source_id → target_id 的有向关系，附带关系类型
│   │          （如 DIRECT_CAUSE、ENABLES、ESCALATES）、因果机制文本描述、
│   │          边超类型（assign_edge_supertypes 填充）和置信度分数。
│   │
│   ├── [C1.3] ThematicLink
│   │          主题边数据类。连接共享同一结构主题的两事件，记录主题名称
│   │          （POWER/WEALTH/KINSHIP/JUSTICE/KNOWLEDGE）、双方参与度
│   │          （direct/indirect）和角色（initiating/enabling 等 7 种）。
│   │
│   ├── [C1.4] Scene
│   │          场景分组数据类。包含 scene_id、events 列表、时空信息和主题标签，
│   │          用于将事件聚类为叙事场景，减少因果分析时的候选对搜索空间。
│   │
│   ├── [C1.5] AgentRole
│   │          Agent 角色数据类。记录角色名（规整后的标准名）、在事件中的角色类型
│   │          （actor/patient/both）和本体分类（如 PROTAGONIST_HERO）。
│   │
│   ├── [C1.6] GenericNode / GenericRelationship
│   │          通用图数据类。用于 graph_mapper 将领域对象（CEKEvent、CausalLink
│   │          等）映射为与特定图数据库/格式无关的抽象节点和边，是导出前的
│   │          统一中间表示。
│   │
│   └── [C1.7] CEKGError / ExtractionError / DAGViolationError
│              自定义异常类层次。CEKGError 为基类，ExtractionError 用于 LLM
│              抽取失败，DAGViolationError 用于因果图环检测违规。
│
├── [C2] 工具函数 · utils.py（254 行）
│   │   提供流水线全局共享的通用工具：异步缓存、DAG 验证、ID 生成、
│   │   字符串安全处理等。
│   │
│   ├── [C2.1] BoundedCache
│   │          线程安全的异步 LRU 缓存类。基于 asyncio.Lock 保证并发安全，
│   │          提供 get/set/clear/size 接口，用哈希键存储 LLM API 响应以
│   │          避免重复调用（同一输入不重复扣费）。
│   │
│   ├── [C2.2] DAGValidator
│   │          有向无环图验证器。维护邻接表，使用 Kahn 算法（拓扑排序）验证
│   │          因果图的 DAG 属性。支持环检测（DFS）、时序约束检查和违规报告。
│   │          导出前和因果链接阶段均调用以确保证因果一致性。
│   │
│   ├── [C2.3] _make_id()
│   │          确定性的 ID 生成器。基于输入字符串的 SHA256 哈希生成短标识符，
│   │          确保同一事件/实体在多次运行中 ID 稳定，支持 checkpoint 恢复。
│   │
│   ├── [C2.4] _hash_for_cache()
│   │          缓存键哈希函数。将 LLM 请求的内容哈希为固定长度字符串，用于
│   │          BoundedCache 的键查找。
│   │
│   └── [C2.5] _escape_cypher_string() / _truncate_safe() / _normalize_weights()
│              Cypher 字符串转义函数（处理反引号和特殊字符）、安全截断函数
│              （按 Unicode 边界截断不破坏多字节字符）、权重归一化函数。
│
└── [C3] 断点续传 · checkpoint_manager.py（291 行）
    │   基于 pickle 的阶段级断点保存/恢复系统。每个流水线阶段完成后自动保存
    │   快照，支持从中断处恢复而不重复执行已完成阶段或消费 API 额度。
    │
    ├── [C3.1] CheckpointManager
    │          断点管理器核心类。管理 checkpoint 目录结构（按 run_id/ 组织），
    │          提供 save(payload, stage) / load(stage) 接口，每次保存附带
    │          SHA256 哈希校验、时间戳和 metadata。支持 JSON 可读副本导出
    │          和进度摘要查询。
    │
    ├── [C3.2] serialize_events() / deserialize_events()
    │          事件对象序列化/反序列化辅助函数。将 CEKEvent 对象列表与 pickle
    │          之间转换，处理字段兼容性和版本迁移。
    │
    └── [C3.3] serialize_links() / deserialize_links()
              链接对象（CausalLink / ThematicLink）序列化/反序列化辅助函数，
              保证 checkpoint 恢复时因果关系链的完整性。
```

### [D] 文本处理与事件抽取层 ─── 从原始文本到结构化事件

```
[D] 文本处理与事件抽取层
│
├── [D1] 文本加载与分割 · text_processor.py（97 行）
│   │   负责原始小说文本的 I/O 和预处理。加载 .txt 文件，剥离 Project Gutenberg
│   │   标准头尾模板（版权声明和元数据），通过多模式正则匹配章节标题边界
│   │   （CHAPTER / Chapter / Book / Part / 罗马数字等 5 种模式），分割为
│   │   结构化章节列表。
│   │
│   ├── [D1.1] split_chapters()
│   │          章节分割核心方法。尝试 5 种正则模式按顺序匹配章节标题，将全文
│   │          切分为 (chapter_title, chapter_text) 元组列表。失败时回退到
│   │          段落级分割（split_into_paragraphs），以段落为最小单元。
│   │
│   ├── [D1.2] load_text()
│   │          文本文件加载函数。处理编码检测（UTF-8 / Latin-1 回退），
│   │          返回纯净文本内容。
│   │
│   └── [D1.3] strip_gutenberg_boilerplate()
│              Project Gutenberg 模板剥离函数。通过模式匹配识别并移除文首的
│              Gutenberg 许可证声明和文末的订阅信息，保留纯叙事内容。
│
├── [D2] LLM 服务层 · llm_service.py（506 行）
│   │   封装所有与 OpenAI/vLLM API 的交互。提供事件抽取、Agent 分类、场景提取、
│   │   因果评估、主题标注五个核心 LLM 能力，统一管理缓存、重试和并发控制。
│   │
│   ├── [D2.1] _async_llm_json_call()
│   │          核心 LLM 异步调用方法。发送 prompt 到 OpenAI/vLLM 端点，
│   │          支持动态 max_tokens 计算（按操作类型自动分配）、JSON 响应
│   │          解析（提取 ```json``` 代码块或裸 JSON）、截断检测告警、
│   │          3 次指数回退重试和缓存命中检查。所有上层 LLM 方法均通过
│   │          此方法发出 API 请求。
│   │
│   ├── [D2.2] extract_events_from_text()
│   │          单段文本事件抽取。向 LLM 注入事件本体论 Prompt（PROMPT_EVENT_
│   │          EXTRACTION），要求 LLM 按本体论分类体系输出 JSON 事件列表。
│   │          返回 CEKEvent 对象列表。
│   │
│   ├── [D2.3] batch_extract_events()
│   │          批量事件抽取的异步编排器。将多个文本段组装为并发 LLM 请求，
│   │          利用 asyncio.gather 并行执行，由 Semaphore 限制最大并发数。
│   │
│   ├── [D2.4] classify_agent_type()
│   │          Agent 类型分类。使用廉价 mini 模型（通过 MINI_MODEL 配置）
│   │          将角色名 + 事件描述分类为 schema 中 47 种 Agent 类型之一，
│   │          如 PROTAGONIST_HERO、MORAL_ANTAGONIST 等。
│   │
│   ├── [D2.5] assess_pairs_bulk()
│   │          批量因果评估。将候选事件对分批发送给 LLM，通过反事实推理
│   │          （"这个效果在没有该原因的情况下会发生吗？"）评估因果关系，
│   │          返回每条候选对的因果判定和置信度。
│   │
│   ├── [D2.6] extract_scenes_from_chapter_async()
│   │          从章节文本异步提取场景分组。LLM 识别文本中的时空断点，
│   │          将事件按地点和时间聚合成 Scene 对象列表。
│   │
│   ├── [D2.7] annotate_single_event_theme()
│   │          单事件主题标注。为一个事件标注 5 大结构主题（POWER/WEALTH/
│   │          KINSHIP/JUSTICE/KNOWLEDGE）的参与程度和叙事角色。
│   │
│   └── [D2.8] init_openai_client() / init_mini_client()
│              OpenAI 客户端初始化函数。分别创建主模型和 mini 模型的
│              AsyncOpenAI 连接实例，支持 OpenAI 和 vLLM 双后端。
│
├── [D3] 指代消解器 · coreference_resolver.py（353 行）
│   │   将 LLM 抽取中出现的代词、昵称、描述词规整为标准角色名。管理角色注册表、
│   │   别名映射和共现矩阵，支持作品特有别名种子（如 Great Expectations 的
│   │   Pip/Mr. Pirrip → Philip Pirrip 映射）。
│   │
│   ├── [D3.1] CoreferenceResolver
│   │          指代消解核心类。维护角色注册表（NameRegistry）和别名映射表，
│   │          提供 resolve(raw_name) 方法将任意名称规整为标准角色名。
│   │          支持子串匹配和名匹配（如 "Elizabeth" → "Elizabeth Bennet"）、
│   │          共现学习（同一事件中出现的角色关系加权）和作品别名种子注入。
│   │
│   ├── [D3.2] get_resolver()
│   │          获取指代消解器单例。与本体管理器类似，首次调用初始化，后续
│   │          调用返回同一实例，保证角色名映射的一致性。
│   │
│   └── [D3.3] KNOWN_WORK_ALIASES
│              作品别名种子字典。按作品名（如 "Great Expectations"）存储
│              已知的角色名-别名映射表，在 Stage0 别名播种阶段注入。
│
└── [D4] RAG 段落索引 · passage_index.py（99 行）
    │   基于 SentenceTransformer 嵌入的段落检索索引。将小说文本按重叠窗口
    │   分割为段落，预计算嵌入向量并 L2 归一化，供因果评估阶段进行上下文
    │   注入时检索最相关的叙事段落。
    │
    ├── [D4.1] PassageIndex
    │         段落索引类。初始化时接收原始文本，调用 _segment_text() 分割
    │         为 300 词重叠 50 词的段落窗口，用 all-MiniLM-L6-v2 模型嵌入，
    │         预归一化后存入内存。提供 search(query, k) 接口做余弦相似度
    │         top-K 检索。
    │
    └── [D4.2] _segment_text()
              文本段落分割函数。按词数滑动窗口（window=300, stride=150）将
              长文本切分为有重叠的段落列表，用于提高检索召回率。
```

### [E] 图谱构建与因果推理层 ─── 从事件列表到因果图

```
[E] 图谱构建与因果推理层
│
├── [E1] 上下文传播 · graph_builder.py（111 行）
│   │   沿事件序列传播上下文属性，实现"叙事持续性"。未显式指定地点/时间的
│   │   后续事件自动继承前驱事件的上下文信息，确保事件图谱的时空完整性。
│   │
│   ├── [E1.1] propagate_context_attributes()
│   │          Pass1 — 地点/时间传播。沿事件链逐事件检查 location 和 time
│   │          字段，若为空则继承紧邻前驱事件的对应属性。保证每个事件至少
│   │          有近似的时空定位。
│   │
│   ├── [E1.2] propagate_context()
│   │          Pass2 — 实体上下文传播。沿事件链传播演员和动机实体（actors +
│   │          why_factors）。规则：新事件显式提及的实体覆盖旧上下文，
│   │          未提及则继承；已从场景消失的角色在一定步数后淡出。
│   │
│   └── [E1.3] _generate_entity_id()
│             规范实体 ID 生成。将角色名/地名/动机因子统一转换为 CEKG 内部
│             使用的实体标识符格式（基于名称的确定性哈希）。
│
├── [E2] 动态上下文候选对 · dynamic_context.py（350 行）
│   │   基于嵌入相似度的因果候选对发现。不评估 O(N²) 全量配对，而是通过
│   │   邻接窗口、场景内嵌入聚类、双滑动窗口长距配对、BM25 关键词匹配和
│   │   实体引导五种策略筛选高质量候选对，上限由 `--max-pairs` 控制。
│   │
│   ├── [E2.1] get_dynamic_context_candidate_pairs()
│   │          策略融合主入口。综合 6 个候选池（邻接对、场景内嵌入、双滑窗
│   │          长距、实体引导、BM25 关键词、局部回退），按优先级分配 max_pairs
│   │          配额，去重后返回最终候选事件对列表。
│   │
│   ├── [E2.2] get_adjacent_pairs()
│   │          邻接对生成。为每个事件取前后 k 个邻居形成滑动窗口内的候选对，
│   │          捕获短程因果链。
│   │
│   ├── [E2.3] get_scene_pairs_by_similarity()
│   │          场景内嵌入配对。在同一场景内的所有事件间计算余弦相似度，
│   │          阈值筛选（由 --thematic-threshold 控制，默认 0.80）后
│   │          返回高语义相关的事件对。
│   │
│   ├── [E2.4] get_long_shot_pairs_double_sliding()
│   │          双滑动窗口长距配对。两个独立窗口在事件序列上滑动，跨窗口
│   │          计算所有事件对的嵌入相似度，发现跨章节的远距离因果关联。
│   │
│   ├── [E2.5] get_bm25_pairs()
│   │          BM25 关键词配对。用 rank-bm25 计算事件文本的关键词重叠度，
│   │          捕捉共享"who/what/why"要素但嵌入相似度可能遗漏的事件对。
│   │
│   ├── [E2.6] _get_embedding_model()
│   │          嵌入模型懒加载。首次调用时初始化 SentenceTransformer
│   │          （all-MiniLM-L6-v2），后续复用同一实例。
│   │
│   └── [E2.7] _cosine_sim() / _encode()
│             余弦相似度计算工具 / 批量嵌入编码工具，为上述策略提供底层
│             向量运算支持。
│
├── [E3] 智能回退链接器 · optimized_linking.py（429 行）
│   │   当 dynamic_context 不可用或 `--no-dynamic-context` 标志启用时，
│   │   作为候选对生成的回退策略。综合实体共现、时间窗口、章节过渡、
│   │   语义相似度和叙事高峰曲线五种策略，各策略独立打分后 Tier 合并。
│   │
│   ├── [E3.1] IntelligentCausalLinker
│   │          智能链接器核心类。管理 5 种候选对生成策略的配置和执行，
│   │          提供 generate_candidate_pairs(events, max_pairs) 主接口。
│   │          每种策略返回 (source_id, target_id, score) 三元组列表，
│   │          最终按 Tier 优先级合并去重并按分数上限截断。
│   │
│   ├── [E3.2] intelligent_long_range_linking()
│   │          长距链接的便捷函数接口。接收事件列表和最大对上限，内部
│   │          实例化 IntelligentCausalLinker 并执行 5 策略融合。
│   │
│   └── [E3.3] EventPair
│              候选对数据类。封装 source_id、target_id、strategy（来源策略
│              名称）和 score（策略内部评分），用于跨策略的去重排序。
│
├── [E4] 因果评估 · integrated_semantic.py（185 行）
│   │   将候选事件对批量送入 LLM 做反事实因果推理。是因果边生成的最后关卡——
│   │   只有 LLM 判定存在因果关系的对才会被转化为 CausalLink 对象写入图。
│   │
│   ├── [E4.1] process_pairs_causal_only()
│   │          因果评估流水线编排。接收候选事件对列表，分批调用
│   │          assess_pairs_causal()（通过 llm_service.assess_pairs_bulk()），
│   │          验证 LLM 返回的因果判定，创建 CausalLink 对象，执行 DAG 合法性
│   │          校验和本体验证后返回有效因果边列表。
│   │
│   └── [E4.2] assess_pairs_causal()
│              因果评估 prompt 构建。将事件对格式化为反事实推理 prompt
│              （McKee 和 Truby 两套因果类型并行注入），调用 LLM 后解析
│              返回的因果类型、机制描述和置信度。
│
└── [E5] 通用图映射 · graph_mapper.py（316 行）
    │   将流水线领域对象（CEKEvent、CausalLink、ThematicLink 等）映射为
    │   与存储格式无关的通用图模型（GenericNode / GenericRelationship），
    │   是导出前的统一中间表示层。
    │
    └── [E5.1] map_to_generic_graph()
              核心映射函数。遍历所有事件、因果边、主题边、Agent、场景、
              地点和动机因子，创建星型图模型——场景为中心节点，关联事件
              和实体为周边节点，扁平化主题标注为可查询节点属性。
```

### [F] 主题标注与导出层 ─── 从因果图到可交付物

```
[F] 主题标注与导出层
│
├── [F1] 主题标注引擎 · theme_annotation.py（466 行）
│   │   为每个事件标注 5 大结构主题（POWER/WEALTH/KINSHIP/JUSTICE/KNOWLEDGE）
│   │   的参与程度，通过 Theme-Bridge 规则将主题信息沿因果链传播，零额外
│   │   API 成本构建 ThematicLink 边。
│   │
│   ├── [F1.1] annotate_event_themes()
│   │          主题标注主入口。执行四步流水线：(a) attach_scene_ids_to_events()
│   │          为事件附加场景 ID；(b) assign_edge_supertypes() 为因果边
│   │          分配 10 种超类型（CAUSAL_PRODUCTION / EMOTIONAL_DRIVE /
│   │          SOCIAL_DYNAMICS 等）；(c) build_local_causal_context() 构建
│   │          局部因果邻域；(d) 逐事件调用 LLM 标注主题，失败事件执行
│   │          两轮重试策略；(e) apply_theme_bridge_rule() 后处理。
│   │
│   ├── [F1.2] apply_theme_bridge_rule()
│   │          Theme-Bridge 桥接规则。若某事件对某主题有 direct 参与度，
│   │          则将因果链上相邻事件对该主题的参与度升级为 indirect+mediating，
│   │          实现主题信息沿叙事因果结构传播。
│   │
│   ├── [F1.3] assign_edge_supertypes()
│   │          边超类型分配。将 McKee/Truby 的细粒度因果类型（92 种）归并
│   │          为 10 种可查询的粗粒度超类型，用于 Graph Explorer 的图例过滤和
│   │          统计分析。
│   │
│   ├── [F1.4] build_local_causal_context()
│   │          局部因果上下文构建。为每个事件汇总其所有因果前驱和后继事件，
│   │          构建因果邻域信息，供 LLM 主题标注时作为叙事上下文注入。
│   │
│   ├── [F1.5] attach_scene_ids_to_events()
│   │          场景 ID 附加。将 Stage5 的场景分组结果关联到对应的 CEKEvent
│   │          对象上，建立事件→场景的归属关系。
│   │
│   └── [F1.6] _clean_theme_annotations()
│              主题标注清理。校验 LLM 返回的主题标注字段完整性，修复
│              缺失项，过滤无效值，确保标注数据符合 schema 定义的格式
│              约束（5 主题 × 参与度等级 × 角色类型）。
│
├── [F2] 主题图构建 · theme_graph.py（132 行）
│   │   从 LLM 主题标注结果中确定性构建 THEMATIC_LINK 边。完全基于标注
│   │   数据，无需额外 API 调用。规则：两事件对同一主题有 direct/indirect
│   │   参与且至少一方为 direct 时建立 THEMATIC 边。
│   │
│   ├── [F2.1] build_thematic_event_edges()
│   │          主题边构建核心函数。遍历所有事件对检查主题共参与条件，
│   │          满足条件的创建 ThematicLink 对象。支持两种模式：causal-projected
│   │          （沿因果边传播主题）和 sequential spine（沿叙事顺序骨架）。
│   │
│   ├── [F2.2] _is_theme_active()
│   │          主题激活判断。检查单事件对指定主题的参与度是否达到 direct
│   │          或 indirect 级别。
│   │
│   └── [F2.3] _safe_conf() / _theme_data()
│             置信度安全获取 / 主题数据字典构建辅助函数。
│
└── [F3] 多格式导出 · exporters.py（553 行）
    │   将处理完成的因果图导出为三种标准格式：JSON-LD（语义网标准）、
    │   Neo4j Cypher 脚本（图数据库导入）、CSV 星型模型（数据分析）。
    │
    ├── [F3.1] build_jsonld()
    │          JSON-LD 导出。构建符合 W3C JSON-LD 规范的图表示，包含
    │          @context 命名空间定义、事件节点（→ CEKEvent 属性）、
    │          因果边（→ CausalLink 属性）和主题边（→ ThematicLink 属性）。
    │          导出文件为 runs/<run>/ge_preprocessed.json。
    │
    ├── [F3.2] export_neo4j_cypher()
    │          Neo4j Cypher 导出。先创建索引和约束，再分批（BATCH_SIZE 行/
    │          文件）生成 MERGE+ON CREATE SET 语句。自动转义标识符中的
    │          反引号和特殊字符，生成可直接导入 Neo4j Browser 的 .cypher 脚本。
    │
    ├── [F3.3] export_csv()
    │          CSV 星型模型导出。生成 13 个 CSV 文件：events.csv（含主题标注）、
    │          agents.csv、acts_in.csv、affected_in.csv、motivates.csv、
    │          places.csv、causes.csv（含因果类型和超类型）、follows.csv、
    │          scenes.csv、thematic_links.csv、generic_nodes.csv、
    │          generic_relationships.csv 和 summary.csv。支持 Neo4j
    │          LOAD CSV 批量导入。
    │
    ├── [F3.4] _escape_cypher_value() / _escape_identifier() / _format_cypher_properties()
    │          Cypher 语法安全转义工具。分别处理属性值转义、标识符转义和
    │          属性字典格式化，防止 Cypher 注入和语法错误。
    │
    └── [F3.5] export_json()
              JSON 导出便捷接口。调用 build_jsonld() 后直接写入文件。
```

---

## 层次总览

| Level | 层级名称 | 数量 | 描述 |
|-------|---------|------|------|
| **L1** | 大模块 | **6** | 流水线宏观阶段 + 前端/后端分层 |
| **L2** | 中模块 | **21** | 每个大模块下的独立文件/子系统 |
| **L3** | 小模块 | **53** | 每个文件内的核心函数/类/数据结构 |

### Level 1（大模块）按流水线数据流排序

```
原始小说 .txt
  │
  ▼
[A] 入口与编排 ──→ [B] 本体与配置
  │                    │
  ▼                    ▼
[D] 文本处理与事件抽取 ←── [C] 数据模型与基础设施
  │
  ▼
[E] 图谱构建与因果推理
  │
  ▼
[F] 主题标注与导出
  │
  ▼
JSON-LD / Neo4j Cypher / CSV
```
