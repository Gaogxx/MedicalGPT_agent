"""本机 .env 读取：已 export 的变量优先、注释与引号处理、.env 不入库。"""
import os
import subprocess
from pathlib import Path

from medagent.envfile import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# 注释\nA_TEST_KEY='abc'\nexport B_TEST_KEY=\"x=y\"\nC_TEST_KEY=\nD_TEST_KEY=new\n", encoding="utf-8")
    for k in ("A_TEST_KEY", "B_TEST_KEY", "C_TEST_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("D_TEST_KEY", "from-shell")
    loaded = load_dotenv(f)
    assert sorted(loaded) == ["A_TEST_KEY", "B_TEST_KEY"]          # 空值不设；返回的是变量名，不含值
    assert os.environ["A_TEST_KEY"] == "abc" and os.environ["B_TEST_KEY"] == "x=y"
    assert os.environ["D_TEST_KEY"] == "from-shell"                 # 终端 export 的优先
    for k in loaded:
        monkeypatch.delenv(k)


def test_dotenv_is_git_ignored_and_template_has_no_secret():
    r = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert r.returncode == 0
    r = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=ROOT)
    assert r.returncode == 1                                        # 模板要入库
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "BAICHUAN_API_KEY=\n" in text and "sk-" not in text
