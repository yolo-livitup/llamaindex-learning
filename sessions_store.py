import json
from datetime import datetime
from pathlib import Path


class SessionStore:
    def __init__(self, sessions_dir="./sessions"):

        self.dir = Path(sessions_dir)
        self.dir.mkdir(exist_ok=True)  # 不存在就创建

    def save(self, session_id, messages):
        """把对话写进 sessions/<id>.json"""
        path = self.dir / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "messages": messages,
        }
        with open(path, "w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    def list(self):
        """列出所有会话（返回文件名列表）"""
        files = sorted(self.dir.glob("*.json"), reverse=True)
        return [f.stem for f in files]

    def load(self, session_id):
        """加载某个会话的 messages"""
        path = self.dir / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["messages"]

    def delete(self, session_id):
        """删除某个会话"""
        path = self.dir / f"{session_id}.json"
        if path.exists():
            path.unlink() #删除单个文件
            return True
        return False

    def new_id(self):
        """生成新会话 ID：20260916_143022"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    session_store = SessionStore()
    sid = session_store.new_id()
    session_store.save(sid,[{"role":"user","content":"测试"}])
    print("会话列表",session_store.list())
    print("加载内容",session_store.load(sid))