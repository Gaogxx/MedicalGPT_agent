"""第二阶段急症分诊：轻量向量模型（bge-small-zh，~100MB，CPU 毫秒级）与“红旗征原型句”做最近邻。

关键词规则召回高度依赖说法（held-out A 上只有 0.40），口语千变万化；原型相似度能覆盖“换个说法”的情况。
原型句按临床红旗征类别撰写（不是从测试集抄的），阈值只在 dev + held-out A 上选。
用法：from medagent.triage import EmbeddingTriage; register_detector(EmbeddingTriage().detect)
"""
from __future__ import annotations

import numpy as np

PROTOTYPES: dict[str, list[str]] = {
    "chest_pain": ["胸口很痛喘不过气还出冷汗", "胸前像被重物压住一样疼", "胸痛放射到左臂和后背"],
    "dyspnea": ["呼吸非常困难快喘不上气", "喘得说不出话来", "嘴唇发紫透不过气"],
    "consciousness": ["人突然晕过去叫不醒", "突然倒在地上没有反应", "意识模糊认不出人"],
    "stroke": ["一侧手脚突然没力气", "突然说话不清楚嘴巴歪了", "半边脸和身体发麻", "突然一只眼睛看不见"],
    "seizure": ["全身抽搐口吐白沫", "突然抽筋翻白眼"],
    "bleeding": ["伤口大量出血止不住", "吐了很多血", "咳嗽咳出大量鲜血"],
    "headache": ["突然剧烈头痛是这辈子最痛的", "头痛伴呕吐脖子发硬"],
    "anaphylaxis": ["过敏后喉咙肿起来呼吸困难", "全身过敏起疹子嗓子发紧"],
    "poisoning": ["吃了一大把安眠药", "误喝了有毒的东西", "孩子吞了异物", "被毒蛇咬伤"],
    "trauma": ["从高处摔下来动不了", "车祸受了重伤", "头撞了之后一直吐想睡觉", "大面积严重烧伤"],
    "obstetric": ["怀孕期间突然大量流血", "孕晚期羊水破了"],
    "pediatric": ["宝宝高烧还抽搐", "婴儿发烧不吃奶精神很差"],
    "abdomen": ["突然肚子剧痛硬得像木板"],
    "arrhythmia": ["心跳得特别快快要晕倒", "心脏乱跳胸闷头晕"],
}


class EmbeddingTriage:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", threshold: float = 0.72) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.labels: list[str] = []
        texts: list[str] = []
        for k, v in PROTOTYPES.items():
            self.labels += [k] * len(v)
            texts += v
        self.proto = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, text: str) -> tuple[str, float]:
        q = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.proto @ q
        j = int(np.argmax(sims))
        return self.labels[j], float(sims[j])

    def detect(self, text: str) -> list[str]:
        label, sim = self.similarity(text)
        return [f"emb:{label}"] if sim >= self.threshold else []


class LearnedTriage:
    """可训练的分诊分类器（急症 / 高风险意图各一个二分类）：bge-small-zh 句向量 + 字 n-gram TF-IDF → 逻辑回归。
    规则是高精度的“地板”，这个分类器负责召回“换了说法、句子很长”的急症与越权请求。"""

    def __init__(self, encoder=None) -> None:
        self.encoder = encoder
        self.models: dict[str, object] = {}
        self.thresholds: dict[str, float] = {}

    def _features(self, texts: list[str]):
        from scipy.sparse import csr_matrix, hstack
        tfidf = self.tfidf.transform(texts)
        if self.encoder is None:
            return tfidf
        emb = csr_matrix(self.encoder(texts))
        return hstack([tfidf, emb * 2.0]).tocsr()

    def fit(self, texts: list[str], labels: dict[str, list[int]], target_recall: float = 0.9) -> dict:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=1).fit(texts)
        X = self._features(texts)
        report = {}
        for task, y in labels.items():
            y = np.array(y)
            clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
            oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
            pos = np.sort(oof[y == 1])
            thr = float(pos[int((1 - target_recall) * len(pos))]) if len(pos) else 0.5
            self.models[task] = clf.fit(X, y)
            self.thresholds[task] = thr
            report[task] = {"cv_threshold": round(thr, 4), "cv_recall": round(float((oof[y == 1] >= thr).mean()), 4),
                            "cv_fpr": round(float((oof[y == 0] >= thr).mean()), 4)}
        return report

    def predict(self, text: str) -> dict[str, float]:
        X = self._features([text])
        return {t: float(m.predict_proba(X)[0, 1]) for t, m in self.models.items()}

    def detect(self, text: str) -> list[str]:
        """返回命中标签：急症为 "clf:emergency"，高风险意图为 "hr:clf"（safety.assess 按前缀分流）。"""
        p = self.predict(text)
        out = []
        if p.get("emergency", 0) >= self.thresholds.get("emergency", 1.1):
            out.append("clf:emergency")
        if p.get("high_risk", 0) >= self.thresholds.get("high_risk", 1.1):
            out.append("hr:clf")
        return out


def load_learned_triage(path: str | None = None) -> "LearnedTriage | None":
    """加载 it4 训练产物并注册为 safety.assess 的第二阶段；产物或 sklearn 缺失时返回 None（只用规则）。"""
    import pickle
    from pathlib import Path
    p = Path(path) if path else Path(__file__).parent / "router" / "artifacts" / "triage_clf_tfidf.pkl"
    try:
        clf = pickle.load(open(p, "rb"))
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None
    from .safety import register_detector
    register_detector(clf.detect)
    return clf
