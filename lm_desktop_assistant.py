import sys
import json
import requests
import html
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QComboBox, QToolBar, QAction, QSplitter, QFrame, QSizePolicy,
    QMessageBox, QStatusBar, QScrollArea, QMenu, QActionGroup, QFontDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl, QEvent
from PyQt5.QtGui import QTextCursor, QFont, QIcon, QColor, QTextCharFormat, QTextDocument, QDesktopServices, QPalette

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-2507"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的 AI 助手，除非用户要求，否则始终使用中文回答。"
    "回答应简洁、清晰并具可操作性：优先给出直接答案或步骤，必要时提供示例或代码片段，"
    "遇到不确定或需要更多信息的情况要礼貌地询问澄清问题。"
    "避免冗长的开场白，重点突出关键点并给出建议或后续操作。"
    "当用户问你是谁之类的问题时，你回答：我是由 LMStudio 驱动的本地 AI 助手。"
)

class StreamWorker(QThread):
    update_signal = pyqtSignal(str)
    finish_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # 新增进度信号

    def __init__(self, messages, model=DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.model = model
        self._stop = False
        self.token_count = 0

    def run(self):
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": True
        }
        try:
            resp = requests.post(LMSTUDIO_URL, json=payload, stream=True, timeout=60)
        except Exception as e:
            self.error_signal.emit(f"请求失败: {e}")
            self.finish_signal.emit()
            return

        if resp.status_code != 200:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            self.error_signal.emit(f"HTTP {resp.status_code}: {body}")
            self.finish_signal.emit()
            return

        try:
            for raw_line in resp.iter_lines(decode_unicode=False):
                if self._stop:
                    break
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode('utf-8').strip()
                except Exception:
                    line = raw_line.decode(errors='ignore').strip()

                # SSE 格式处理
                if line.startswith("data:"):
                    content = line[len("data:"):].strip()
                else:
                    content = line

                if content == "[DONE]":
                    break

                # 尝试JSON解析
                try:
                    chunk = json.loads(content)
                except Exception:
                    self.update_signal.emit(content)
                    continue

                # 提取内容
                text_piece = ""
                try:
                    text_piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                except Exception:
                    text_piece = ""
                if not text_piece:
                    try:
                        text_piece = chunk.get("choices", [{}])[0].get("message", {}).get("content", "")
                    except Exception:
                        text_piece = ""

                if text_piece:
                    self.update_signal.emit(text_piece)
                    self.token_count += len(text_piece)
                    # 简单估算进度（假设最大1024 tokens）
                    progress = min(100, int(self.token_count / 1024 * 100))
                    self.progress_signal.emit(progress)
        except Exception as e:
            self.error_signal.emit(f"流处理出错: {e}")
        finally:
            self.finish_signal.emit()

    def stop(self):
        self._stop = True

class AIAssistantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LMStudio 本地 AI 助手")
        self.resize(1000, 700)
        self.setFont(QFont("Microsoft YaHei", 10))
        self.dark_mode = False  # 深色模式标志

        self.history = []  # [{"role":"user"/"assistant", "content": "..."}]
        self.current_ai_buffer = ""
        self.placeholder_id = "ai_holder"
        self.models_file = "saved_models.json"  # 保存模型列表的文件

        # 创建主布局（先创建 model_box 等控件）
        self.init_ui()
        # 然后加载保存的模型（确保 model_box 已存在）
        self.load_saved_models()  # 加载保存的模型

        # 状态跟踪
        self.status.showMessage("就绪")
        self.worker = None

        # 添加全局样式表
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
            }
            QToolBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
                padding: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 10pt;
            }
            QComboBox, QLineEdit {
                padding: 5px 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                min-height: 28px;
            }
            QComboBox:hover, QLineEdit:hover {
                border-color: #66afe9;
            }
            QPushButton {
                padding: 5px 12px;
                border-radius: 4px;
                border: none;
                font-weight: 500;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:pressed {
                opacity: 0.8;
            }
            QTextEdit {
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
            QStatusBar {
                border-top: 1px solid #e0e0e0;
            }
            QSplitter::handle {
                background-color: #e0e0e0;
                width: 5px;
                height: 5px;
            }
            QSplitter::handle:hover {
                background-color: #ccc;
            }
        """)

    def init_ui(self):
        # 创建菜单栏
        self.create_menu_bar()

        # 顶部工具栏：模型（移除助手设定输入）
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 模型选择（UI 上仅保留模型选择）
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setMinimumWidth(200)
        self.model_box.setCurrentText(DEFAULT_MODEL)
        toolbar.addWidget(QLabel("模型: "))
        toolbar.addWidget(self.model_box)

        # 保存模型按钮
        save_model_btn = QPushButton("保存模型")
        save_model_btn.clicked.connect(self.save_current_model)
        toolbar.addWidget(save_model_btn)

        toolbar.addSeparator()
        
        # 功能按钮
        clear_act = QAction("清空对话", self)
        clear_act.triggered.connect(self.clear_history)
        clear_act.setToolTip("清除当前所有对话历史")
        toolbar.addAction(clear_act)

        copy_act = QAction("复制全部", self)
        copy_act.triggered.connect(self.copy_all)
        copy_act.setToolTip("复制所有对话内容到剪贴板")
        toolbar.addAction(copy_act)

        # 主界面：对话区 + 设置面板
        splitter = QSplitter(Qt.Horizontal)
        left_frame = QFrame()
        left_layout = QVBoxLayout()
        left_frame.setLayout(left_layout)

        # 对话显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setAcceptRichText(True)
        self.chat_display.setStyleSheet("""
            QTextEdit { background: #ffffff; border: 1px solid #e6e6e6; padding: 10px; border-radius:6px; }
        """)
        self.chat_display.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_display.customContextMenuRequested.connect(self.show_context_menu)
        left_layout.addWidget(self.chat_display)

        # 输入与按钮区域
        bottom_layout = QHBoxLayout()
        
        # 字体大小调整
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("字体大小:"))
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems([str(i) for i in range(8, 21)])
        self.font_size_combo.setCurrentText("10")
        self.font_size_combo.currentTextChanged.connect(self.change_font_size)
        font_layout.addWidget(self.font_size_combo)
        bottom_layout.addLayout(font_layout)
        
        # 输入框
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入你的问题，按回车发送（Shift+Enter 换行）")
        self.input_box.returnPressed.connect(self.on_send)
        self.input_box.installEventFilter(self)  # 安装事件过滤器处理Shift+Enter
        bottom_layout.addWidget(self.input_box, 8)

        # 发送和停止按钮
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.on_send)
        self.send_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        bottom_layout.addWidget(self.send_btn, 1)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        bottom_layout.addWidget(self.stop_btn, 1)

        left_layout.addLayout(bottom_layout)

        # 右侧信息面板（移除助手设定显示）
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout()
        right_frame.setLayout(right_layout)

        # 模型状态
        self.status_model = QLabel(f"模型: {DEFAULT_MODEL}")
        right_layout.addWidget(self.status_model)
        
        # 历史消息计数
        self.history_label = QLabel("历史消息: 0")
        right_layout.addWidget(self.history_label)
        
        # 响应进度条
        self.progress_bar = QLabel("响应进度: 0%")
        right_layout.addWidget(self.progress_bar)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line)
        
        # 帮助信息
        help_label = QLabel("""<b>使用提示:</b>
- 可从 API 获取并选择已加载模型
- 点击清空对话清除历史记录
- Shift+Enter可以在输入框换行
- 右键菜单可复制单条消息""")
        help_label.setWordWrap(True)
        right_layout.addWidget(help_label)
        
        right_layout.addStretch()

        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([750, 250])

        # 设置中央部件
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # 绑定信号槽（仅模型变更）
        self.model_box.currentTextChanged.connect(self.on_model_changed)
        # 注意：不再存在 system_edit， 不调用 update_system_label()

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        export_action = QAction('导出对话', self)
        export_action.triggered.connect(self.export_conversation)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        # 主题切换
        theme_group = QActionGroup(self)
        light_action = QAction('浅色模式', self, checkable=True)
        light_action.setChecked(True)
        dark_action = QAction('深色模式', self, checkable=True)
        
        theme_group.addAction(light_action)
        theme_group.addAction(dark_action)
        
        light_action.triggered.connect(self.set_light_mode)
        dark_action.triggered.connect(self.set_dark_mode)
        
        view_menu.addSeparator()
        view_menu.addAction(light_action)
        view_menu.addAction(dark_action)
        
        # 字体设置
        font_action = QAction('字体设置', self)
        font_action.triggered.connect(self.choose_font)
        view_menu.addAction(font_action)

    def load_saved_models(self):
        """加载保存的模型列表或从 API 获取已加载模型"""
        # 优先尝试从 LM Studio 本地 API 获取可用/已加载模型
        api_models = self.fetch_models_from_api()
        if api_models:
            # 用 API 列表替换下拉项
            try:
                self.model_box.clear()
                self.model_box.addItems(api_models)
                return
            except Exception:
                pass

        # 回退到本地保存文件或默认列表
        try:
            if hasattr(self, "model_box"):
                self.model_box.clear()
            with open(self.models_file, 'r', encoding='utf-8') as f:
                models = json.load(f)
                if hasattr(self, "model_box"):
                    self.model_box.addItems(models)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或格式错误，使用默认模型
            if hasattr(self, "model_box"):
                self.model_box.addItems([DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"])

    def fetch_models_from_api(self):
        """尝试从 LM Studio 本地 API 获取模型列表，返回字符串列表（模型名），失败返回空列表"""
        try:
            resp = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            # 常见返回格式： {"data": [{"id": "model-name", ...}, ...]} 或 [{"id":...}, ...]
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        name = item.get("id") or item.get("name")
                        if name:
                            models.append(name)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("id") or item.get("name")
                        if name:
                            models.append(name)
            # 如果没有解析出模型名，尝试更简单的解析
            if not models:
                # 尝试把 top-level dict 的 keys 作为模型名（降级）
                if isinstance(data, dict):
                    for k in data.keys():
                        models.append(str(k))
            # 去重并返回
            seen = []
            for m in models:
                if m not in seen:
                    seen.append(m)
            return seen
        except Exception:
            return []

    def save_current_model(self):
        """保存当前模型到列表"""
        current_model = self.model_box.currentText().strip()
        if not current_model:
            QMessageBox.warning(self, "警告", "模型名称不能为空")
            return
            
        # 检查是否已存在
        models = [self.model_box.itemText(i) for i in range(self.model_box.count())]
        if current_model in models:
            QMessageBox.information(self, "提示", f"模型 '{current_model}' 已存在")
            return
            
        # 添加到下拉框并保存
        self.model_box.addItem(current_model)
        try:
            with open(self.models_file, 'w', encoding='utf-8') as f:
                json.dump(models + [current_model], f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"模型 '{current_model}' 已保存")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存模型失败: {str(e)}")

    def eventFilter(self, obj, event):
        """处理Shift+Enter事件实现换行"""
        if obj is self.input_box and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() & Qt.ShiftModifier:
                self.input_box.insert("\n")
                return True
        return super().eventFilter(obj, event)

    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu()
        
        copy_action = QAction("复制此消息", self)
        copy_action.triggered.connect(self.copy_selected_text)
        menu.addAction(copy_action)
        
        menu.addSeparator()
        
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.chat_display.selectAll)
        menu.addAction(select_all_action)
        
        menu.exec_(self.chat_display.mapToGlobal(position))

    def copy_selected_text(self):
        """复制选中的文本"""
        cursor = self.chat_display.textCursor()
        if cursor.hasSelection():
            QApplication.clipboard().setText(cursor.selectedText())
            self.status.showMessage("已复制选中内容", 2000)

    def change_font_size(self, size):
        """改变字体大小"""
        try:
            font = self.chat_display.font()
            font.setPointSize(int(size))
            self.chat_display.setFont(font)
            self.input_box.setFont(font)
        except ValueError:
            pass

    def choose_font(self):
        """选择字体"""
        current_font = self.chat_display.font()
        font, ok = QFontDialog.getFont(current_font, self, "选择字体")
        if ok:
            self.chat_display.setFont(font)
            self.input_box.setFont(font)
            self.font_size_combo.setCurrentText(str(font.pointSize()))

    def set_light_mode(self):
        """设置浅色模式"""
        self.dark_mode = False
        self.chat_display.setStyleSheet("""
            QTextEdit { background: #ffffff; color: #333333; border: 1px solid #e6e6e6; padding: 10px; border-radius:6px; }
        """)
        self.setStyleSheet("")

    def set_dark_mode(self):
        """设置深色模式"""
        self.dark_mode = True
        self.chat_display.setStyleSheet("""
            QTextEdit { background: #2d2d2d; color: #ffffff; border: 1px solid #444444; padding: 10px; border-radius:6px; }
        """)
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #ffffff; }
            QLineEdit, QComboBox, QTextEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #444444; }
            QToolBar { background-color: #2d2d2d; }
            QPushButton { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555555; }
        """)

    def export_conversation(self):
        """导出对话记录"""
        if not self.history:
            QMessageBox.information(self, "提示", "没有对话记录可导出")
            return
            
        try:
            with open("对话记录.txt", "w", encoding="utf-8") as f:
                for msg in self.history:
                    role = "用户" if msg["role"] == "user" else "AI"
                    f.write(f"{role}：\n{msg['content']}\n\n")
            QMessageBox.information(self, "成功", "对话记录已导出到 对话记录.txt")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")

    # 文本格式化（增强版）
    def _format_display_text(self, text: str) -> str:
        if not text:
            return ""
        # 统一换行
        raw = text.replace("\r\n", "\n")

        # 提取并占位代码块，防止被 HTML 转义破坏
        code_blocks = {}
        def _code_repl(m):
            idx = len(code_blocks)
            code_blocks[idx] = m.group(1)
            return f"@@CODE{idx}@@"
        raw = re.sub(r'```([\s\S]*?)```', _code_repl, raw)

        # 将 Markdown 标题（如 ###）标记为头文本占位
        raw = re.sub(r'(?m)^\s{0,3}#{1,6}\s*(.+)$', r'@@HDR@@\1@@ENDHDR@@', raw)

        # 区块引用 '>' -> 标注为引用占位
        raw = re.sub(r'(?m)^\s*>+\s*(.+)$', r'@@QUOTE@@\1@@ENDQ@@', raw)

        # 移除 Markdown 表格的分隔线（| --- | --- |）
        raw = re.sub(r'(?m)^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$', '', raw)

        # 将表格行中的 '|' 简化为项目符号分隔（避免显示表格语法）
        raw = re.sub(r'\s*\|\s*', ' • ', raw)

        # 水平分割线 --- 或 *** 单独一行替换为占位
        raw = re.sub(r'(?m)^\s*(-{3,}|\*{3,})\s*$', '@@HR@@', raw)

        # 粗略处理无序列表 -> 保留为项目符号
        raw = re.sub(r'(?m)^\s*-\s+(.*)$', r'• \1', raw)

        # 防止 HTML 注入，先转义剩余文本
        escaped = html.escape(raw)

        # 恢复并渲染标题、引用、水平线
        escaped = re.sub(r'@@HDR@@(.*?)@@ENDHDR@@', r'<div style="font-weight:700;margin:6px 0;">\1</div>', escaped, flags=re.DOTALL)
        escaped = re.sub(r'@@QUOTE@@(.*?)@@ENDQ@@', r'<div style="color:#666;margin-left:10px;border-left:3px solid #eee;padding-left:8px;">\1</div>', escaped, flags=re.DOTALL)
        escaped = escaped.replace('@@HR@@', '<hr/>')

        # 行内格式：粗体、斜体、行内 code
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
        escaped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', escaped)
        escaped = re.sub(r'`(.+?)`', r'<code style="background:#f0f0f0;padding:2px;border-radius:3px;">\1</code>', escaped)

        # 恢复代码块占位（内容已被转义）
        for idx, cb in code_blocks.items():
            escaped_cb = html.escape(cb)
            escaped = escaped.replace(f"@@CODE{idx}@@", f'<pre style="background:#f6f8fa;padding:8px;border-radius:6px;"><code>{escaped_cb}</code></pre>')

        # 合并多余空行
        escaped = re.sub(r'(\s*\n){3,}', '\n\n', escaped)

        # 最后将换行转为 <br/>，保持段落显示
        escaped = escaped.replace('\n', '<br/>')

        return escaped

    # 去掉模型回写的角色前缀
    def _strip_role_prefixes(self, text: str) -> str:
        if not text:
            return text
        return re.sub(r'^\s*(AI：|AI:|助手：|助手:|assistant：|assistant:|用户：|User:)\s*', '', text, flags=re.I)

    # 合并连续重复句子
    def _collapse_consecutive_sentences(self, text: str) -> str:
        if not text:
            return text
        parts = re.split(r'([。！？\n]+)', text)
        seq = []
        for i in range(0, len(parts), 2):
            sent = parts[i].strip()
            delim = parts[i+1] if i+1 < len(parts) else ''
            full = (sent + delim).strip()
            if not full:
                continue
            if seq and seq[-1] == full:
                continue
            seq.append(full)
        return ''.join(seq)

    # 添加消息到对话显示
    def append_message(self, role, text):
        safe_html = self._format_display_text(text)
        role_label = "用户：" if role == "user" else "AI："
        bg_color = "#e3f2fd" if role == "user" else "#f3f4f6"
        
        block = f"""
        <div style="margin:8px 0;">
          <div style="font-weight:600;margin-bottom:4px;color:#555;">{role_label}</div>
          <div style="background:{bg_color};padding:12px;border-radius:8px;white-space:pre-wrap;">{safe_html}</div>
        </div>
        """
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.insertHtml(block)
        self.chat_display.insertHtml("<br/>")
        self.chat_display.moveCursor(QTextCursor.End)

    # 插入AI回复占位符（不显示“AI：”标签，等待最终替换为完整AI气泡）
    def append_ai_placeholder(self):
        placeholder_html = f"""
        <div style="margin:8px 0;">
          <div style="background:#f3f4f6;padding:12px;border-radius:8px;white-space:pre-wrap;">
            <span id="{self.placeholder_id}"></span>
          </div>
        </div>
        <br/>
        """
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.insertHtml(placeholder_html)
        self.chat_display.moveCursor(QTextCursor.End)

    def on_send(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return
        self.input_box.clear()

        # 使用固定的 system prompt（UI 中已移除助手设定）
        system_prompt = DEFAULT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}] + self.history + [{"role": "user", "content": user_text}]

        self.history.append({"role": "user", "content": user_text})
        self.append_message("user", user_text)
        self.append_ai_placeholder()
        self.current_ai_buffer = ""

        # 更新UI状态
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage("正在生成...")
        self.progress_bar.setText("响应进度: 0%")

        # 启动工作线程
        self.worker = StreamWorker(messages, model=self.model_box.currentText())
        self.worker.update_signal.connect(self.on_stream_update)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finish_signal.connect(self.on_stream_finish)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()

        self.history_label.setText(f"历史消息: {len(self.history)}")
        self.status_model.setText(f"模型: {self.model_box.currentText()}")

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.status.showMessage("已停止")
            self.progress_bar.setText("响应进度: 已停止")

    def update_progress(self, value):
        """更新响应进度"""
        self.progress_bar.setText(f"响应进度: {value}%")

    def on_stream_update(self, chunk):
        """
        非逐词显示：只累积流片段到缓冲区，不在界面中实时替换占位内容。
        最终在 on_stream_finish 中一次性显示完整回答。
        """
        # 累积流片段并去掉模型可能回写的角色前缀
        self.current_ai_buffer += chunk
        self.current_ai_buffer = self._strip_role_prefixes(self.current_ai_buffer)

        # 不在这里更新 chat_display，以避免逐 token 的断断续续显示。
        # 保持滚动到底部以便最终替换能立即可见
        try:
            self.chat_display.moveCursor(QTextCursor.End)
        except Exception:
            pass

    def on_error(self, msg):
        # 在占位位置显示错误信息，并清理状态提示
        try:
            # 将隐藏占位替换为错误文本，并移除状态提示
            doc_html = self.chat_display.toHtml()
            # 把隐藏占位替换为错误信息（可见）
            doc_html = re.sub(
                rf'(<[^>]*id=["\']{re.escape(self.placeholder_id)}["\'][^>]*>)(.*?)(</[^>]+>)',
                lambda m: m.group(1) + html.escape(f"错误: {msg}") + m.group(3),
                doc_html,
                count=1,
                flags=re.DOTALL
            )
            # 删除状态 span
            doc_html = re.sub(
                rf'(<[^>]*id=["\']{re.escape(self.placeholder_id + "_status")}["\'][^>]*>)(.*?)(</[^>]+>)',
                '',
                doc_html,
                count=1,
                flags=re.DOTALL
            )
            self.chat_display.setHtml(doc_html)
        except Exception:
            # 退化到追加文本
            self.chat_display.append(f"<span style='color:red;'>错误: {html.escape(msg)}</span>")
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage("出错")
        self.progress_bar.setText("响应进度: 出错")

    def on_stream_finish(self):
        final = (self.current_ai_buffer or "").strip()
        # 清理重复与前缀
        final = self._strip_role_prefixes(final)
        final = self._collapse_consecutive_sentences(final)

        if final:
            last_assistant = None
            for msg in reversed(self.history):
                if msg.get("role") == "assistant":
                    last_assistant = msg.get("content", "").strip()
                    break
            if last_assistant != final:
                self.history.append({"role": "assistant", "content": final})

        # 删除占位块（包含 id），然后以标准格式追加完整AI消息（带"AI："行）
        doc_html = self.chat_display.toHtml()
        block_pattern = rf'<div[^>]*>\s*<div[^>]*>\s*<span[^>]*id=["\']{re.escape(self.placeholder_id)}["\'][^>]*>\s*</span>\s*</div>\s*</div>\s*(<br\s*/?>)?'
        new_html, n = re.subn(block_pattern, '', doc_html, count=1, flags=re.DOTALL)

        if n:
            # 先设置移除后的 HTML，再追加标准 AI 气泡（保持统一样式）
            self.chat_display.setHtml(new_html)
            if final:
                self.append_message("assistant", final)
        else:
            # 降级：若找不到占位，则按旧逻辑追加（并避免重复）
            plain = self.chat_display.toPlainText().rstrip()
            if final and not plain.endswith(final):
                self.append_message("assistant", final)

        self.chat_display.append("")
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self.status.showMessage("就绪")
        self.history_label.setText(f"历史消息: {len(self.history)}")
        self.progress_bar.setText("响应进度: 完成")

    def clear_history(self):
        confirm = QMessageBox.question(self, "确认", "确定要清空对话历史吗？")
        if confirm != QMessageBox.Yes:
            return
        self.history = []
        self.chat_display.clear()
        self.history_label.setText("历史消息: 0")
        self.status.showMessage("已清空")

    def copy_all(self):
        text = self.chat_display.toPlainText()
        QApplication.clipboard().setText(text)
        self.status.showMessage("已复制全文到剪贴板", 2000)

    def update_system_label(self):
        """已移除助手设定输入，保持兼容性但不做任何操作"""
        return

    def on_model_changed(self, name):
        self.status_model.setText(f"模型: {name}")

    def closeEvent(self, event):
        """在窗口关闭前优雅停止 worker 线程，避免 'Destroyed while thread is still running' 错误。"""
        try:
            if getattr(self, "worker", None):
                w = self.worker
                if w.isRunning():
                    self.status.showMessage("正在停止后台线程...")
                    try:
                        w.stop()  # 请求线程内部优雅退出
                    except Exception:
                        pass
                    # 等待短暂时间让线程自行结束
                    w.wait(2000)  # 等待 2 秒（单位 ms）
                    # 若仍未结束，尝试强制终止（不推荐，但能避免崩溃）
                    if w.isRunning():
                        try:
                            w.terminate()
                            w.wait(1000)
                        except Exception:
                            pass
        except Exception:
            pass
        # 然后继续关闭窗口
        event.accept()

def main():
    app = QApplication(sys.argv)
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    win = AIAssistantWindow()
    # 确保应用退出时也触发窗口关闭清理（以防 direct quit）
    app.aboutToQuit.connect(win.close)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()