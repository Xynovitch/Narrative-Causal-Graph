# Causal Graph(0326) Feedback

## Thematic Layer

<aside>

실제로 구현된 내용

</aside>

- 5개의 Theme (POWER / WEALTH / KINSHIP / JUSTICE / KNOWLEDGE)이 코드 및 실제 결과물에 일관되게 반영되어 있다. (theme_annotation.py, llm_service.py)
- 각 event를 대상으로 5개 Theme에 대한 annotation object가 생성된다.
- Annotation 과정에서 event 정보, actor/patient, scene_id, 인접 causal 관계를 포함한 local context를 사용하고, 이후 Theme-Bridge Rule까지 적용된다. (CEKG Thematic Layer v2.pdf, pipeline.py, theme_annotation.json)
- Thematic layer가 구조적으로 구현된 증거: 모든 vent에 대해 5개 theme가 포함되고, 각 theme마다 involvement, role, evidence, signals, confidence 필드를 갖는 구조가 확인된다. (theme_annotation.json)

<aside>

한계점?

</aside>

- **confidence 값:** confidence 값이 비어 있거나, 근거를 설명하기 어려워서 theme annotation의 신뢰도를 비교하거나 필터링하는 데에 한계가 있다.
- **Theme-Bridge Rule 이후 annotation :** 일부 theme에서 annotation이 불완전하다. 예를 들어서 event/597439eb는 KINSHIP, WEALTH만 evidence가 없는데도 indirect relation으로 표기된다. 즉, indirect thematic relation의 근거를 어떻게 설명할지 고민할 필요가 있다.
- **Theme 과잉 태깅 가능성:** 일부 event에서 여러 theme(ex. POWER와 KNOWLEDGE)이 동시에 활성화되는 패턴이 반복적으로 나타난다. 특히 KNOWLEDGE(“replies”, “asks”, “swear” 같은 언어/정보 행위), POWER(“sternly asks” 같은 authority tone)와 같이 비교적 넓은 개념을 가진 theme은 핵심 causal mechanism이 아닌 약한 신호에도 쉽게 반응하여 direct, indirect로 확장된다. 본래 CEKG Thematic Layer의 목적과 달리, 실제로는 해당 event의 core mechanism이 아닌, secondary signal인데도 theme이 활성화되는 경우가 있다.
- **role 활용 부족:** null, mediating 중심으로 편향될 가능성이 있다. 예를 들어서 event/597439eb는 JUSTICE에서 direct involvement이지만 role이 없다. role은 thematic chain 내에서 initiating, enabling과 같은 기능적 역할인데, 현재 그 역할이 제대로 효력을 발휘하지 못하고 있다.

```
    "event/597439eb": {
      "JUSTICE": {
        "involvement": "direct",
        "role": null,
        "evidence": "Mike nervously replies about the man's preparedness to swear.",
        "signals": [
          "nervously replies",
          "preparedness to swear"
        ],
        "confidence": null
      },
      "KNOWLEDGE": {
        "involvement": "direct",
        "role": null,
        "evidence": "Mike nervously replies about the man's preparedness to swear.",
        "signals": [
          "nervously replies",
          "preparedness to swear"
        ],
        "confidence": null
      },
      "KINSHIP": {
        "involvement": "indirect",
        "role": "mediating",
        "evidence": "",
        "signals": [],
        "confidence": null
      },
      "POWER": {
        "involvement": "indirect",
        "role": null,
        "evidence": "Mr. Jaggers sternly asks Mike what the man is prepared to swear.",
        "signals": [
          "sternly asks",
          "prepared to swear"
        ],
        "confidence": null
      },
      "WEALTH": {
        "involvement": "none",
        "role": null,
        "evidence": "",
        "signals": [],
        "confidence": null
      }
```

<aside>

개선 방안 (논의해 볼 것)

</aside>

- annotation의 방향성과 신뢰도, 세부 사항을 조정할 필요성.

## Causal Relation Edge

<aside>
💡

한계점?

</aside>

- RelationType 종류는 다양하지만, dictionary만큼 다양하게 분포되지 않고 일부 relation이 반복적으로 등장한다.
- EVENT_ENABLES_NEXT
- EVENT_REINFORCEMENT
- COMPASSION_TRIGGER
- ALLY_DEPENDENCE
- EMOTIONAL_MANIPULATION
- CAUSES_REVERSAL

<aside>
💡

해결 방안?

</aside>

- relation ontology를 더 다양하게 활용할 수 있는 방안 강구
- relation label의 precision을 정량적으로 검증할 방안?

## **Example Sub-graph: Hidden Wealth / False Expectations**

It shows the novel’s main causal twist in a way that both scholars and casual readers can grasp immediately. It captures the book’s most important structural claim:

> Pip’s rise looks like a fairy-tale social ascent, but is actually caused by a hidden source of money, and the whole identity-structure collapses when that source is revealed.
> 

### Focus on the chain from:

- Pip helping Magwitch in the marshes
- Pip’s shame after meeting Estella
- Pip’s desire to become a gentleman
- Jaggers announcing Pip’s “great expectations”
- Pip wrongly assuming Miss Havisham is his benefactor
- Magwitch’s return and revelation
- Pip’s horror and reinterpretation of his whole life

### Broad event sequence

1. Pip helps the convict
2. Pip meets Estella and feels ashamed
3. Pip desires gentility
4. Pip is told of his expectations
5. Pip assumes Miss Havisham is the source
6. Pip builds his future around Estella
7. Magwitch returns
8. Magwitch reveals he is the benefactor
9. Pip’s expectations collapse
10. Pip reinterprets his rise

### 1. Pip helping Magwitch in the marshes

- `event/a30e9de2`
    
    **The man asked Pip to get him a file and wittles.**
    
    chapter 1, sequence 16
    
- `event/361affd3`
    
    **I said that I would get him the file, and I would get him what broken bits of food I could, and I would come to him at the Battery, early in the morning.**
    
    chapter 1, sequence 22
    
- `event/8e794986`
    
    **I stole some bread, some rind of cheese, about half a jar of mincemeat, some brandy, a meat bone, and a beautiful round compact pork pie.**
    
    chapter 2, sequence 71
    
- `event/d351d1bd`
    
    **I unlocked and unbolted the door in the kitchen, communicating with the forge, and got a file from among Joe’s tools.**
    
    chapter 2, sequence 72
    
- `event/56f3e378`
    
    **Pip handed the man the file and he laid it down on the grass.**
    
    chapter 3, sequence 82
    

---

### 2. Pip’s shame after meeting Estella

- `event/82dee4b0`
    
    **Estella expressed disdain for Philip Pirrip's use of 'Jacks' instead of 'Knaves' and criticized his hands and boots.**
    
    chapter 8, sequence 365
    
- `event/1e236a8a`
    
    **Philip Pirrip began to feel ashamed of his hands due to Estella's contempt.**
    
    chapter 8, sequence 366
    
- `event/53dbccf5`
    
    **Estella won the card game against Philip Pirrip and called him a stupid, clumsy labouring-boy.**
    
    chapter 8, sequence 367
    

---

### 3. Pip desires gentility

- `event/2c2c2146`
    
    **Pip told Biddy he wanted to be a gentleman.**
    
    chapter 17, sequence 829
    
- `event/99ce2366`
    
    **I answered, 'The beautiful young lady at Miss Havisham’s ... I want to be a gentleman on her account.'**
    
    chapter 17, sequence 835
    

---

### 4. Jaggers announcing Pip’s “great expectations”

- `event/aaf1a0fa`
    
    **Mr. Jaggers communicates to Pip that he has great expectations.**
    
    chapter 18, sequence 901
    
- `event/187f318a`
    
    **Mr. Jaggers informs Pip that he will come into a handsome property and be brought up as a gentleman.**
    
    chapter 18, sequence 902
    
- `event/0bdfd709`
    
    **Mr. Jaggers informs Pip that the name of his benefactor remains a secret.**
    
    chapter 18, sequence 904
    

---

### 5. Pip wrongly assuming Miss Havisham is the source

- `event/918d0e34`
    
    **Pip expressed his previous belief that Miss Havisham was the benefactor.**
    
    chapter 40, sequence 2246
    
- `event/7e80efee`
    
    **Estella paused in her knitting and perceived Pip had discovered his real benefactor.**
    
    chapter 44, sequence 2393
    
- `event/299a255d`
    
    **Pip revealed he found out who his patron is.**
    
    chapter 44, sequence 2397
    
- `event/fd27223f`
    
    **Miss Havisham confirmed Pip's assumption about his initial visits.**
    
    chapter 44, sequence 2398
    

---

### 6. Pip builds his future around Estella

- `event/99ce2366`
    
    **...I admire her dreadfully, and I want to be a gentleman on her account.**
    
- `event/4f2654c1`
    
    **Philip Pirrip confessed his love to Estella.**
    
    chapter 44, sequence 2413
    

---

### 7. Magwitch returns / reveals he is the benefactor

- `event/ddeda708`
    
    **He said, 'Yes, Pip, dear boy, I’ve made a gentleman on you! It’s me wot has done it!'**
    
    chapter 39, sequence 2159
    
- `event/6b1390b2`
    
    **The man reveals he swore to make Pip rich and a gentleman.**
    
    chapter 39, sequence 2160
    
- `event/899fdce7`
    
    **The man claims to be Pip's second father and has put away money for him.**
    
    chapter 39, sequence 2161
    
- `event/2005ae8b`
    
    **Mr. Jaggers confirmed that Abel Magwitch is the benefactor.**
    
    chapter 40, sequence 2245
    

---

### 8. Pip’s expectations collapse / reinterpretation

- `event/5ac1867c`
    
    **I am heavily in debt,—very heavily for me, who have now no expectations,—and I have been bred to no calling, and I am fit for nothing.**
    
    chapter 41, sequence 2282
    
- `event/299a255d`
    
    **Pip revealed he found out who his patron is.**
    
- `event/94195f61`
    
    **My repugnance to Magwitch had all melted away.**
    
    chapter 54, sequence 3065
    

## 제안

현재 그래프에서 탐색하기 좋은 theme_annotation + 인물(agent) 필터링이 잘 안 된다. → 데이터를 질문하기 좋은 형태로 정규화해서 접근성을 높여야 한다.

### 1) `theme_annotations`가 구조화된 속성이 아니라 “문자열(JSON text)”로 들어가 있음 (csv 문제)

CSV를 업로드할 경우 Neo4j가 `theme_annotations`를 텍스트 덩어리(e. g. `"WEALTH": {"involvement": "direct", ...}`)로 가지고 있어 긴 문자열 안에 특정 문구가 들어 있는지만 `CONTAINS`로 찾고 있음.

- 공백/줄바꿈/인코딩/따옴표 형태가 다르면 안 잡힘
- direct/indirect만 따로 안정적으로 추출하기 어려움
- `WEALTH + KNOWLEDGE` 같은 복합 필터가 매우 불안정해짐

---

#### 2) agent 필터가 “의미적으로 연결된 중심 인물”이 아니라, 현재 edge가 붙은 event만 잡고 있음

예를 들어, 현재 query는 `Pip + WEALTH`를 

> Pip이 `ACTS_IN` 또는 `AFFECTED_IN`으로 직접 연결된 event 중에서, WEALTH annotation이 있는 것
> 

으로 읽어냄.

그런데 소설 상에서는 Pip과 관련된 장면이어도,

- agent linking 누락
- Pip이 직접 행위자/피해자로 tagging되지 않음
- 대명사 혹은 다른 명칭 (e.g. pip/philip pirrip, magwitch/convict/unknown man)으로 표기될 경우, 다른 인물로 집계되어서 일일이 지정해야 함.

등의 원인으로 Pip이라는 인물 중심의 그래프를 탐색하기 어려움

**→ agent tagging 개선 필요: LLM이 causal relation을 재확인하는 과정에서, 작품별 agent dictionary를 지정하고 결합하는 절차가 추가되어야 함.** 

---

## 3) event segmentation이 잘게 쪼개져 있어서, 해석 단위와 그래프 단위가 다름

CEKG의 event는 미세하게 쪼개져 있다. 예를 들어서 Pip helps Magwitch라는, 해석적으로 하나의 plot-unit이 그래프 안에서 미시 사건 (e.g. convict asks for file / Pip promises / Pip steals food . . . )으로 분산되어 있다. 그래서 어떤 노드에는 WEALTH가, 어떤 노드에는 Pip agent가 붙는 등 필터링에 방해가 된다. 

→ “Pip helping Magwitch cluster”와 같이, 하나의 scene-level 안에서 sequence of events를 clustering하거나, scene으로 cluster를 구성하여 미시 event를 묶어서 결과물을 출력한다.

---