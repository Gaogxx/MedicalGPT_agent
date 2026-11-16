"""禁忌症交叉检查（2026-11-16）：回答里推荐 / 提到的药，对照用户描述的病情（问题 + 历史）是否属于禁忌。

为什么：最终 9B 的漏网失败里有“脑出血的辅助治疗 → 建议阿替普酶溶栓”这类事实性用药错误，Laya（编码器分类模型）
看不出来，规则可以：药名和病情都能确定性地匹配，命中就升级强模型带着证据重答。

两个来源：
  1. 内置精选 KB（kb.json）的 contraindications 字段（只有 10 个药有）；
  2. 下面手工整理的高风险“药物—病情”禁忌表：只收教科书级、无争议的组合（溶栓 / 抗凝遇出血、喹诺酮 / 四环素遇儿童或孕妇、
     ACEI / ARB / 他汀遇妊娠、阿司匹林遇儿童病毒感染……）。与 KB 一样未经医生审核，命中只用于“升级强模型重答”，不直接改写回答。
扩展 KB（Wikidata，2,483 个药）不含禁忌信息，不参与。

“提到”≠“推荐”：药名所在的那一小段里有“禁用 / 不能 / 避免 / 慎用 / 不建议……”时视为提醒，是“是否服用过……”时视为问诊，都不算命中。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# 只认否定 / 警示；“遵医嘱”“在医生指导下使用”是附加提醒，不算没推荐（“可用阿替普酶溶栓，需遵医嘱”正是要拦的）
_NEG = re.compile(r"禁用|禁忌|不能|不可|不宜|不要|不应|避免|慎用|勿|不建议|不推荐|切勿|严禁|停用|停药|不适合|不适用")
# 否定 / 问诊只看药名所在的那一小段（按逗号切）：“如辛伐他汀和阿替普酶，但具体用药必须由医生……切勿自行用药”里，
# “切勿”否定的是自行用药，不是这个药（2026-11-16 在真实回答上漏掉的就是这一句）
_CLAUSE = re.compile(r"[。！!？?\n；;，,]")
_ASK = re.compile(r"是否|有没有|有无|是不是|吗|请问|多久了")       # “是否长期服用阿司匹林”是问诊，不是推荐
# 年龄前面不能是数字或“十”：“五十六岁”“16岁”不是儿童
_AGE = r"(?<![0-9十几一二三四五六七八九])(?:[1-9]|十[一二]?|[一二三四五六七八九])岁"
_CHILD = r"宝宝|婴儿|婴幼儿|新生儿|幼儿|儿童|小孩|孩子|小朋友|女儿|儿子|" + _AGE
_PREG = r"怀孕|孕妇|孕期|妊娠|备孕|孕早期|孕中期|孕晚期|怀了"


@dataclass(frozen=True)
class Rule:
    drug: str          # 药名正则（回答里找）
    condition: str     # 病情正则（问题 / 历史里找）
    reason: str
    also: str = ""     # 可选：病情还需同时满足的第二个条件（如“儿童”且“病毒感染”）


RULES: tuple[Rule, ...] = (
    Rule(r"阿替普酶|尿激酶|链激酶|瑞替普酶|替奈普酶|溶栓", r"脑出血|颅内出血|出血性(脑)?卒中|蛛网膜下腔出血|消化道出血|活动性出血",
         "溶栓药禁用于颅内出血 / 活动性出血"),
    # 不含肝素：脑出血后用低分子肝素预防深静脉血栓是指南认可的做法
    Rule(r"华法林|利伐沙班|达比加群|阿哌沙班|阿司匹林|氯吡格雷|替格瑞洛", r"脑出血|颅内出血|出血性(脑)?卒中|蛛网膜下腔出血|消化道出血|活动性出血|血友病",
         "出血性疾病禁用抗凝 / 抗血小板药"),
    Rule(r"阿司匹林", _CHILD, "儿童病毒感染（发热、流感、水痘）用阿司匹林有瑞氏综合征风险", also=r"发热|发烧|流感|感冒|水痘|病毒"),
    Rule(r"[左莫环诺氧加]氟?沙星|沙星类|喹诺酮", _CHILD + "|" + _PREG + r"|哺乳", "喹诺酮类不宜用于 18 岁以下、孕妇及哺乳期"),
    Rule(r"四环素|多西环素|米诺环素|土霉素",
         r"宝宝|婴儿|婴幼儿|新生儿|幼儿|(?<![0-9十几一二三四五六七八九])(?:[1-7]|[一二三四五六七])岁|" + _PREG,
         "四环素类禁用于 8 岁以下儿童和孕妇（牙齿着色、骨发育）"),
    Rule(r"卡托普利|依那普利|贝那普利|培哚普利|雷米普利|福辛普利|赖诺普利|咪达普利|普利类|"
         r"氯沙坦|缬沙坦|厄贝沙坦|替米沙坦|坎地沙坦|奥美沙坦|阿利沙坦|沙坦类", _PREG, "ACEI / ARB 类降压药妊娠期禁用（致畸）"),
    Rule(r"阿托伐他汀|瑞舒伐他汀|辛伐他汀|普伐他汀|洛伐他汀|氟伐他汀|匹伐他汀|他汀类", _PREG, "他汀类妊娠期禁用"),
    Rule(r"华法林", _PREG, "华法林妊娠期禁用（致畸）"),
    Rule(r"布洛芬|双氯芬酸|萘普生|吲哚美辛|塞来昔布|美洛昔康|阿司匹林", r"胃溃疡|十二指肠溃疡|消化性溃疡|消化道溃疡|消化道出血|胃出血",
         "消化性溃疡 / 消化道出血慎用或禁用非甾体抗炎药"),
    Rule(r"普萘洛尔|心得安", r"哮喘", "非选择性 β 受体阻滞剂禁用于支气管哮喘"),
    Rule(r"硝酸甘油|单硝酸异山梨酯|硝酸异山梨酯|硝酸酯", r"西地那非|伟哥|他达拉非|万艾可", "硝酸酯类与 PDE5 抑制剂合用可致严重低血压"),
    Rule(r"二甲双胍", r"严重肾功能不全|肾衰|尿毒症|透析|酮症酸中毒", "二甲双胍禁用于严重肾功能不全、酮症酸中毒"),
)


@lru_cache(maxsize=1)
def _kb_rules() -> tuple[Rule, ...]:
    """把精选 KB 的禁忌字段（“严重肾功能不全、酮症酸中毒、缺氧性疾病禁用”）拆成病情词。"""
    from ..knowledge import load_kb
    out = []
    for name, d in load_kb().drugs.items():
        text = d.get("contraindications") or ""
        terms = [re.sub(r"(禁用|慎用|患者|者)$", "", t.strip()) for t in re.split(r"[、，,；;。及和]", text)]
        terms = [t for t in terms if len(t) >= 3]
        if terms:
            out.append(Rule(re.escape(name), "|".join(map(re.escape, terms)), f"{name}：{text}"))
    return tuple(out)


def check(answer: str, context: str) -> list[str]:
    """返回命中的证据（“脑出血 + 阿替普酶：溶栓药禁用于颅内出血 / 活动性出血”）；context = 历史 + 问题。"""
    hits: list[str] = []
    sents = [s for s in _CLAUSE.split(answer or "") if s.strip()]
    for rule in RULES + _kb_rules():
        cond = re.search(rule.condition, context)
        if not cond or (rule.also and not re.search(rule.also, context)):
            continue
        for s in sents:
            m = re.search(rule.drug, s)
            if m and not _NEG.search(s) and not _ASK.search(s):
                hits.append(f"{cond.group(0)} + {m.group(0)}：{rule.reason}")
                break
    return list(dict.fromkeys(hits))
