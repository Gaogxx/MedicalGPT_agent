/**
 * dsh-medical-tools：医疗工具插件（DeepSeek Harness 真实 API）
 *
 * 用 defineTool 注册 3 个医疗工具：check_symptom / search_drug / search_guideline。
 * MVP 阶段使用内置规则知识库，后续可替换为真实医学知识图谱 / 药典数据库。
 */
import type { Context } from '../../vendor/cordis/src/index.ts'
import { defineTool } from '../../packages/core/tools/src/index.ts'

export const name = 'dsh-medical-tools'
export const inject = ['tools']

// ============ 症状数据库 ============
const SYMPTOM_DB: Record<string, {
  possible_causes: string[]
  recommendation: string
  urgency: string
  confidence: number
}> = {
  '头痛': { possible_causes: ['偏头痛', '紧张性头痛', '高血压'], recommendation: '建议休息观察，如持续超过3天或加重请就医', urgency: 'medium', confidence: 0.65 },
  '发烧': { possible_causes: ['流行性感冒', '病毒性感染', '细菌性感染'], recommendation: '多饮水休息，体温超过38.5℃可遵医嘱退烧，持续高热需就医', urgency: 'medium', confidence: 0.7 },
  '咳嗽': { possible_causes: ['上呼吸道感染', '支气管炎', '过敏性咳嗽'], recommendation: '注意休息多饮水，持续超过2周请就医行胸片检查', urgency: 'low', confidence: 0.55 },
  '腹泻': { possible_causes: ['急性胃肠炎', '消化不良', '食物不耐受'], recommendation: '注意补充水分和电解质，频繁腹泻或便血请及时就医', urgency: 'medium', confidence: 0.6 },
  '胃痛': { possible_causes: ['胃炎', '消化性溃疡', '功能性消化不良'], recommendation: '避免刺激性食物，规律饮食，持续疼痛或黑便请就医', urgency: 'medium', confidence: 0.55 },
  '胸痛': { possible_causes: ['心绞痛', '心肌梗死', '胸膜炎'], recommendation: '胸痛可能为急症，请立即就医或拨打120', urgency: 'emergency', confidence: 0.75 },
  '呼吸困难': { possible_causes: ['哮喘发作', '心力衰竭', '肺栓塞'], recommendation: '呼吸困难为急症，请立即拨打120', urgency: 'emergency', confidence: 0.8 },
  '关节痛': { possible_causes: ['骨关节炎', '类风湿关节炎', '痛风'], recommendation: '避免过度活动，可就医行风湿免疫相关检查', urgency: 'low', confidence: 0.5 },
  '头晕': { possible_causes: ['体位性低血压', '贫血', '前庭功能紊乱'], recommendation: '避免突然站起，持续头晕请就医检查血压和血常规', urgency: 'medium', confidence: 0.5 },
  '恶心': { possible_causes: ['胃肠功能紊乱', '消化不良', '妊娠反应'], recommendation: '清淡饮食，注意休息，持续恶心呕吐请就医', urgency: 'low', confidence: 0.5 },
}

// ============ 药品数据库 ============
const DRUG_DB: Record<string, Record<string, any>> = {
  '布洛芬': { name: '布洛芬', generic_name: 'Ibuprofen', indications: '缓解轻至中度疼痛，如头痛、关节痛、牙痛、痛经', contraindications: '对本品过敏者禁用，孕妇禁用，严重肝肾功能不全者禁用', dosage: '成人一次1片（0.2g），一日3次，饭后服用', side_effects: '胃肠道不适、恶心、头晕、皮疹', interactions: ['阿司匹林', '华法林', '甲氨蝶呤'], pregnancy_category: 'C' },
  '对乙酰氨基酚': { name: '对乙酰氨基酚', generic_name: 'Paracetamol', indications: '解热镇痛，用于感冒发热、头痛、关节痛等', contraindications: '严重肝肾功能不全者禁用，对本品过敏者禁用', dosage: '成人一次0.5g，一日不超过2g', side_effects: '偶见皮疹、恶心，过量可致肝损伤', interactions: ['华法林'], pregnancy_category: 'B' },
  '阿司匹林': { name: '阿司匹林', generic_name: 'Aspirin', indications: '解热镇痛、抗血小板聚集', contraindications: '活动性消化道溃疡出血者禁用，对本品过敏者禁用，儿童病毒感染时禁用', dosage: '遵医嘱，抗血小板常用75-100mg/日', side_effects: '胃肠道刺激、出血风险、过敏反应', interactions: ['布洛芬', '华法林', '甲氨蝶呤'], pregnancy_category: 'D' },
  '阿莫西林': { name: '阿莫西林', generic_name: 'Amoxicillin', indications: '敏感菌所致的呼吸道、泌尿道等感染', contraindications: '青霉素过敏者禁用', dosage: '遵医嘱，成人常用0.5g，一日3次', side_effects: '过敏反应、胃肠道不适、皮疹', interactions: ['华法林'], pregnancy_category: 'B' },
  '二甲双胍': { name: '二甲双胍', generic_name: 'Metformin', indications: '2型糖尿病一线降糖药', contraindications: '严重肾功能不全、酮症酸中毒、缺氧性疾病禁用', dosage: '遵医嘱，成人起始0.5g，一日2次', side_effects: '胃肠道反应、乳酸酸中毒（罕见）', interactions: [], pregnancy_category: 'B' },
}

// ============ 指南数据库 ============
const GUIDELINE_DB: Record<string, Record<string, any>> = {
  '高血压': { topic: '高血压', content: '低盐饮食（每日食盐<5g）、控制体重、戒烟限酒、增加蔬果和钾摄入、减少饱和脂肪；规律服药并监测血压。', source: '《中国高血压防治指南》', year: 2023 },
  '糖尿病': { topic: '2型糖尿病', content: '二甲双胍为一线首选降糖药；合并动脉粥样硬化性心血管疾病或高风险者可优先选择SGLT2抑制剂或GLP-1受体激动剂。', source: '《中国2型糖尿病防治指南》', year: 2020 },
  '高血脂': { topic: '高血脂', content: '生活方式干预（低脂饮食、运动、控制体重）为基础；他汀类药物为降脂首选。', source: '《中国血脂管理指南》', year: 2023 },
  '痛风': { topic: '痛风', content: '限制高嘌呤食物（动物内脏、海鲜、啤酒），多饮水，急性期使用非甾体抗炎药或秋水仙碱，缓解期降尿酸治疗。', source: '《中国高尿酸血症与痛风诊疗指南》', year: 2019 },
}

// ============ 工具逻辑 ============
function analyzeSymptom(symptom: string, duration?: string): Record<string, any> {
  for (const [key, info] of Object.entries(SYMPTOM_DB)) {
    if (symptom.includes(key)) {
      const result: Record<string, any> = { ...info, symptom }
      if (duration) result.duration = duration
      return result
    }
  }
  return {
    possible_causes: ['暂无法判断'],
    recommendation: '症状信息不足，建议前往医院就诊进行详细检查',
    urgency: 'medium',
    confidence: 0.3,
    symptom,
    ...(duration ? { duration } : {}),
  }
}

// 症状/适应症关键词，用于 search_drug 的模糊反查
const SYMPTOM_KEYWORDS = [
  '头痛', '发烧', '发热', '退烧', '咳嗽', '腹泻', '胃痛', '关节痛', '痛经', '牙痛',
  '胸痛', '呼吸困难', '嗓子疼', '喉咙', '恶心', '头晕', '感冒', '疼痛', '止痛',
  '降糖', '消炎', '感染', '发热', '嗓子',
]

function searchDrug(name: string): Record<string, any> | null {
  const trimmed = name.trim()
  // 1. 精确药名匹配
  if (DRUG_DB[trimmed]) return DRUG_DB[trimmed]
  // 2. 药名子串匹配（如「布洛芬片」→「布洛芬」）
  for (const [drugName, info] of Object.entries(DRUG_DB)) {
    if (trimmed.includes(drugName) || drugName.includes(trimmed)) {
      return { ...info, matched: `已匹配药品「${drugName}」` }
    }
  }
  // 3. 症状/适应症关键词反查（如「头痛」「止痛药」「退烧」→ 相关药品）
  const hitKeywords = SYMPTOM_KEYWORDS.filter((kw) => trimmed.includes(kw))
  if (hitKeywords.length > 0) {
    const related = Object.entries(DRUG_DB)
      .filter(([, info]) => hitKeywords.some((kw) => (info.indications as string).includes(kw)))
      .map(([dn]) => dn)
    if (related.length > 0) {
      return {
        query: name,
        matched_by: hitKeywords[0],
        related_drugs: related,
        suggestion: `「${name}」不是具体药名。针对「${hitKeywords[0]}」可参考的药品：${related.join('、')}。请用具体药名查询用法、副作用等详情。`,
      }
    }
  }
  return null
}

function searchGuideline(topic: string, aspect?: string): Record<string, any> | null {
  for (const [key, val] of Object.entries(GUIDELINE_DB)) {
    if (topic.includes(key)) {
      const result: Record<string, any> = { ...val }
      if (aspect) result.aspect = aspect
      return result
    }
  }
  return null
}

// ============ 插件入口 ============
export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'check_symptom',
    description: '根据患者描述的症状分析可能原因，提供初步建议',
    parameters: {
      symptom: { type: 'string', required: true, description: '患者描述的症状' },
      duration: { type: 'string', description: '症状持续时间，如"3天"、"1周"' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    async execute(args) {
      return analyzeSymptom(args.symptom, args.duration)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'search_drug',
    description: '查询药品的适应症、禁忌症、用法用量、副作用等信息。name 必须是具体药品名（如「布洛芬」），不要传症状或疾病名；如果只知道症状不知道药名，请先用 check_symptom 分析症状。',
    parameters: {
      name: { type: 'string', required: true, description: '药品名称' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    async execute(args) {
      const result = searchDrug(args.name)
      if (result === null) {
        return { error: `未找到药品「${args.name}」的信息，请确认药品名称是否正确。` }
      }
      return result
    },
  }))

  ctx.tools.register(defineTool({
    name: 'search_guideline',
    description: '检索临床指南中关于某疾病或主题的推荐意见',
    parameters: {
      topic: { type: 'string', required: true, description: '疾病或主题' },
      aspect: { type: 'string', description: '关注的方面' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    async execute(args) {
      const result = searchGuideline(args.topic, args.aspect)
      if (result === null) {
        return { error: `未找到关于「${args.topic}」的指南信息。` }
      }
      return result
    },
  }))
}
