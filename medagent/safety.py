"""医疗安全三关：输入预检（紧急 / 危机 / 高风险意图）→ 工具前检查 → 输出后处理。

紧急识别不走模型、不走路由：确定性规则命中后立即给出急救提示（P0 路径，延迟≈0），
再把后续解释交给强模型。规则词表在 dev 集（原 80 条用例）上调，在 held-out 集上报告。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (类别, 正则)。按类别聚合是为了审计时能说清“为什么判紧急”。
_EMERGENCY: list[tuple[str, str]] = [
    # it3：按临床红旗征（red flags）分族泛化——“疼/痛”互换、允许 2~4 字间隔、口语说法；只用 dev + held-out A 调参
    ("chest_pain", r"胸(口|部)?.{0,3}(剧|撕裂|刀割|压榨)?.{0,2}(痛|疼)|心口.{0,3}(痛|疼|压|闷)|胸(口)?(闷|痛|疼).{0,8}(冷汗|出汗|大汗|胳膊|左臂|后背)|"
                   r"(像|被).{0,4}(石头|大石|东西)压着|压榨(样|感)|撕裂(样|一样|般)?.{0,2}(痛|疼)"),
    ("dyspnea", r"呼吸困难|喘不(上|过)气|窒息|憋.{0,4}(发紫|喘|呼吸|不过气)|呼吸.{0,3}(急促|很急|困难)|(嘴唇|口唇|脸).{0,3}(发紫|发绀|青紫)|"
                r"上不来气|喘得.{0,6}(说不|讲不)|(哮喘|喘).{0,8}(不管用|没用|缓解不了)"),
    ("consciousness", r"意识(不清|模糊|丧失)|昏迷|(突然)?(晕|昏)(倒|过去|厥)|叫不醒|不省人事|失去知觉|倒地(不起)?|迷迷糊糊.{0,4}(叫|醒)?"),
    ("stroke", r"说不出话|说话.{0,3}(含糊|不清)|口齿不清|(口角|嘴角?|嘴巴).{0,4}(歪|斜)|半(边|侧)?(身|身子|脸).{0,4}(麻|无力|不能动|动不了|不遂)|"
               r"一(侧|边|只)(肢体|手脚|身体|手|胳膊|腿)?.{0,4}(麻木|无力|抬不起|动不了)|突然.{0,6}(看不见|失明|重影)"),
    ("seizure", r"抽搐|抽风|癫痫发作|四肢抽动"),
    ("bleeding", r"大出血|(血).{0,6}(止不住|按不住|往外冒)|吐血|呕血|咯血|咳(出|了).{0,4}(鲜)?血|大量(便血|出血)|血流不止|流了(很|好)多血"),
    ("thunderclap_headache", r"(剧烈|爆炸样|炸裂(样|般)?)(的)?头痛|头痛.{0,6}(炸开|欲裂)|头痛.{0,6}(喷射(状|性)?呕吐|呕吐)|"
                             r"(最痛|最严重)的(一次)?头痛|头痛.{0,10}(最痛|最严重)|头痛.{0,6}脖子(发)?硬|颈(部)?(僵硬|发硬)"),
    ("head_injury", r"(撞|磕|摔|碰)(到|伤|了).{0,3}头.{0,10}(吐|呕|困|嗜睡|昏)"),
    ("anaphylaxis", r"过敏性休克|喉(头|咙)?.{0,2}(水肿|发紧|肿)|(嘴唇|舌头|喉咙)肿.{0,6}(呼吸|喘|说不出)"),
    ("poisoning", r"中毒|误服|误吞|喝(了)?.{0,4}农药|吞(了|下).{0,6}(药|电池|异物)|一氧化碳|煤气|(一整瓶|一大把|很多片|整盒).{0,4}(药|安眠药)|"
                  r"(吃|喝)了.{0,4}(一整瓶|一大把|整盒)|(蛇|毒蛇).{0,2}咬"),
    ("arrhythmia", r"心跳.{0,4}(特别|非常|极其|异常)(快|乱)|心(率|跳).{0,6}(1[5-9]\d|2\d\d)|心慌.{0,6}(胸痛|晕|喘|冷汗)|心脏跳得.{0,4}(乱|快).{0,6}晕"),
    ("trauma", r"车祸|(高处)?坠落|从.{0,4}(楼|高处).{0,3}(摔|掉|跳)下|骨头(都)?(外露|露出)|(大面积|一大片).{0,2}烧(伤|烫伤)|烫伤(了)?(一大片|大面积)|"
               r"刀伤.{0,4}(很深|流血不止)|腰.{0,2}动不了"),
    ("acute_abdomen", r"(剧烈|突然).{0,4}(腹痛|肚子痛).{0,10}(硬|板)|肚子硬得像(木板|石头)|板状腹"),
    ("obstetric", r"(怀孕|孕期|孕妇|孕\d+周).{0,10}(出血|流血|流了.{0,3}血|剧烈腹痛|胎动消失)|羊水破"),
    ("pediatric_fever", r"(婴儿|宝宝|新生儿|孩子).{0,8}(高烧|发烧|发热).{0,8}(抽|不吃奶|嗜睡|精神差)"),
]
# it4（事后修补，见 README）：独立集 C 中 3/3 条自伤危机消息全部漏检，扩充口语表达
_CRISIS = (r"自杀|轻生|自残|不想(再)?活|活(着|下去)(真|都)?没(啥|什么)?意思|活不下去|结束(自己的)?生命|割腕|跳楼|"
           r"想(从.{0,6})?(往下)?跳(下去)?|楼顶.{0,6}跳|攒了.{0,6}(安眠药|药)|(全|都)吃了就这样|划了?.{0,6}(好几道|手腕|胳膊)|"
           r"不如死了|死了算了|不想再活下去")

_HIGH_RISK: list[tuple[str, str]] = [
    # “开”前后都要有限定，避免“开心，吃了药”这类误报
    ("prescribe", r"(给我|帮我|替我|能不能|可不可以|请你?)开.{0,6}(处方|药|抗生素|激素|针|方子)|开(一?个|一?点|一?些|一?张)(处方|.{0,4}(药|抗生素|激素|方子))|处方|"
                  r"(直接)?告诉我.{0,4}(吃|用)(什么|哪种|几片|多少).{0,4}(药|抗生素|能好)?"),
    ("diagnose", r"(帮我|给我|替我|能不能|可以)(诊断|确诊)|得了什么病|是不是(得了)?.{0,4}(癌|肿瘤|白血病|肝炎|恶性|良性)|"
                 r"判断.{0,10}(是不是|是否).{0,6}(癌|病|恶性|良性)"),
    ("surgery", r"(做|该做|需要做)(什么)?手术|要不要(开刀|手术)"),
    ("treatment_plan", r"治疗方案|怎么治(疗)?最好|(推荐|给我)(一个|个)?治疗|(给我|帮我)?(定|制定|出)(个|一个)?.{0,6}方案"),
    ("admission", r"(是不是|要不要|需不需要|需要)住院"),
    ("dose_change", r"(剂量|药量).{0,4}(加倍|翻倍|减半)|能不能(自己)?停药|(自行|自己).{0,3}(停|减|加|换)(了|药|量|半)?|"
                    r"(加倍|翻倍|减半)(吃|用|打)|(换|改)(个|一种).{0,4}(更好|别)的.{0,4}药"),
]
_DISCLAIMER_HINT = re.compile(r"仅供参考|不能替代|不能代替|请?(及时|尽快)?就医|遵医嘱|咨询(专业)?(医生|药师)|以医生(的)?(诊断|意见)为准")
_EMERGENCY_ADVICE = re.compile(r"120|急救|急诊|立即就医|马上就医|立刻就医|尽快就医")
PRESCRIPTION_PATTERN = re.compile(
    r"(\d+(\.\d+)?\s*(mg|g|μg|毫克|克|片|粒|袋|ml|毫升|单位)).{0,10}(每日|一日|一天|每天|每次|一次|每晚|tid|bid|qd|q\d+h)",
    re.I)

HIGH_RISK_DEFERRAL = "处方、诊断、手术与治疗方案需由执业医师面诊后决定，我无法替您开具或判断；可以帮您整理症状、准备就诊时要问的问题。"
DISCLAIMER = "（以上信息仅供健康科普参考，不能替代执业医师的诊断与治疗；如症状持续或加重请及时就医。）"
EMERGENCY_BANNER = "⚠️ 您描述的情况可能是急症，请立即拨打 120 或尽快前往最近医院急诊，不要等待线上回复。"
CRISIS_BANNER = ("⚠️ 听起来您现在很难受。如果有伤害自己的想法，请立即联系身边的人，或拨打 120 / 心理援助热线"
                 "（全国心理援助热线 400-161-9995 或当地 12356），您不是一个人。")


_EXTRA_DETECTORS: list = []   # 可插拔第二阶段（如 triage.EmbeddingTriage），返回命中类别列表


def register_detector(fn) -> None:
    _EXTRA_DETECTORS.append(fn)


@dataclass
class SafetyAssessment:
    level: str = "normal"                  # emergency | normal
    crisis: bool = False
    high_risk: bool = False
    emergency_hits: list[str] = field(default_factory=list)
    high_risk_hits: list[str] = field(default_factory=list)

    @property
    def requires_strong_model(self) -> bool:
        return self.level == "emergency" or self.crisis or self.high_risk


def _hits(text: str, table: list[tuple[str, str]]) -> list[str]:
    return [name for name, pat in table if re.search(pat, text)]


def current_turn(text: str) -> str:
    """评测集把长历史拼在问题前（“我之前问过…。 现在…”）；安全判定只看当前这一句，否则历史里的“呼吸困难”会误报。"""
    m = re.match(r"^我之前问过.+?。\s*(.+)$", text)
    return m.group(1) if m else text


def assess(text: str) -> SafetyAssessment:
    text = current_turn(text)
    em = _hits(text, _EMERGENCY)
    extra_hr: list[str] = []
    for extra in _EXTRA_DETECTORS:
        for h in extra(text):
            if h.startswith("hr:"):
                extra_hr.append(h)
            elif h not in em:
                em.append(h)
    # “止不住”单独出现时要求同句有“血”，避免“咳嗽止不住”误报
    if em == ["bleeding"] and "血" not in text:
        em = []
    hr = _hits(text, _HIGH_RISK) + extra_hr
    return SafetyAssessment(level="emergency" if em else "normal", crisis=bool(re.search(_CRISIS, text)),
                            high_risk=bool(hr), emergency_hits=em, high_risk_hits=hr)


def has_disclaimer(text: str) -> bool:
    return bool(_DISCLAIMER_HINT.search(text))


def has_emergency_advice(text: str) -> bool:
    return bool(_EMERGENCY_ADVICE.search(text))


def contains_prescription(text: str) -> bool:
    return bool(PRESCRIPTION_PATTERN.search(text))


def guard_output(answer: str, assessment: SafetyAssessment) -> str:
    """输出关：急症 / 危机前置横幅；高风险请求里出现“剂量+频次”的处方式表述时改写；兜底免责声明。"""
    out = answer.strip()
    if assessment.crisis and CRISIS_BANNER not in out:
        out = f"{CRISIS_BANNER}\n\n{out}"
    if assessment.level == "emergency" and not has_emergency_advice(out[:120]):
        out = f"{EMERGENCY_BANNER}\n\n{out}"
    if assessment.high_risk:
        if contains_prescription(out):
            out = PRESCRIPTION_PATTERN.sub("（具体剂量请遵医嘱）", out)
        if HIGH_RISK_DEFERRAL not in out:
            out += f"\n\n{HIGH_RISK_DEFERRAL}"
    if not has_disclaimer(out):
        out = f"{out}\n\n{DISCLAIMER}"
    return out
